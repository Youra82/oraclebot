# src/oraclebot/model/transformer.py
# Transformer-Architektur (Spezifikation Schritt 6): kontinuierliche Feature-Projektion
# -> pro-Timeframe-Encoder -> Multi-Timeframe-Attention-Fusion
# -> stufenweiser Decoder (Trend->Range->Form->Struktur) -> Beam Search.
#
# Frueher wurden die Feature-Vektoren erst per K-Means auf diskrete Token-IDs abgebildet
# (siehe tokenizer.py) und dann per nn.Embedding in den Modellraum projiziert -- das
# entspricht der Sprachmodell-Analogie (diskretes Vokabular), verliert bei kontinuierlichen
# Finanzsignalen aber echte Information (zwei leicht unterschiedliche RSI-Werte koennen im
# selben Cluster landen). Jetzt wird der standardisierte Feature-Vektor direkt per
# nn.Linear in den Modellraum projiziert (wie bei einem Vision Transformer) -- keine
# Diskretisierung, keine Informationsverlust durch Clustering.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from oraclebot.data.targets import TARGET_NAMES

TARGET_CARDINALITIES = {
    'trend': 2,            # bearish/bullish -- keine Neutral-Klasse mehr (siehe targets.py)
    'range': 4,            # 0-0.5 / 0.5-1 / 1-2 / >2 ATR
    'close_position': 3,   # lower/middle/upper third
    'upper_wick': 3,        # small/medium/large
    'lower_wick': 3,        # small/medium/large
    'gap_yn': 2,            # kein/nennenswerter Gap zum Vortages-Close
    'inside_outside_day': 3,  # normal / inside day / outside day
    'high_first': 2,          # Tagestief zuerst erreicht / Tageshoch zuerst erreicht
}


def estimate_attention_memory_bytes(window_sizes: dict, timeframes: list, batch_size: int,
                                     nhead: int, num_layers: int, d_model: int = 0,
                                     dim_feedforward: int = 0, dtype_bytes: int = 4) -> int:
    """Grobe Abschaetzung des Spitzenspeichers ueber ALLE Timeframe-Encoder zusammen

    (die waehrend eines Forward+Backward-Schritts gleichzeitig im Speicher gehalten werden,
    weil PyTorch's Autograd alle Zwischenaktivierungen fuers Backward retained).

    Zwei Terme pro Layer und Timeframe:
    1) O(L^2)-Attention-Scores-Matrix: batch * nhead * (seq_len+1)^2 * dtype_bytes
    2) O(L*d_model)-Aktivierungen (Q/K/V-Projektionen, Attention-Output, FFN-Hidden/Output):
       batch * (seq_len+1) * (~6*d_model + dim_feedforward) * dtype_bytes
       Bei kleinem d_model (z.B. 32) ist Term 2 vernachlaessigbar gegenueber Term 1; bei
       groesserem d_model (z.B. 256) wird er vergleichbar gross und darf nicht mehr wegfallen.

    Diese Funktion existiert, weil ein erster Trainingsversuch mit batch_size=32 und
    seq_len=2000 (15m-Fenster) den Trainings-PC dreimal zum Einfrieren/Neustarten gezwungen
    hat -- RAM-Erschoepfung durch quadratisch wachsenden Attention-Speicher, nicht CPU-Last
    oder Hitze. Siehe Memory-Eintrag feedback_cpu_memory_safety.
    """
    total = 0
    for tf in timeframes:
        seq_len_with_cls = window_sizes[tf] + 1
        attention_term = batch_size * nhead * (seq_len_with_cls ** 2) * dtype_bytes
        activation_term = batch_size * seq_len_with_cls * (6 * d_model + dim_feedforward) * dtype_bytes
        total += (attention_term + activation_term) * num_layers
    return total


class PositionalEncoding(nn.Module):
    """Klassische sinusfoermige Positionskodierung (Vaswani et al.), fest (nicht gelernt)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimeframeEncoder(nn.Module):
    """Kodiert die Feature-Sequenz EINES Timeframes zu einem einzelnen Kontext-Vektor.

    Ein gelerntes CLS-Token wird vorangestellt (wie bei BERT); seine Ausgabe nach dem
    Transformer-Encoder ist die Zusammenfassung der gesamten Sequenz.
    """

    def __init__(self, n_features: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, max_len: int):
        super().__init__()
        self.feature_projection = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, seq_len, n_features) Float -> (batch, d_model)"""
        batch_size = features.size(0)
        x = self.feature_projection(features)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        return x[:, 0]


class TimeframeFusion(nn.Module):
    """Multi-Timeframe Attention (Spezifikation Punkt 6): fusioniert die Kontext-Vektoren

    aller Timeframes ueber eine Attention-Schicht, in der jeder Timeframe-Kontext ein
    "Token" ist. Die Attention-Gewichte des CLS-Tokens auf die einzelnen Timeframes
    entsprechen direkt der "welcher Timeframe ist gerade wichtig"-Idee aus der Spezifikation
    und werden fuer Interpretierbarkeit mit zurueckgegeben.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float, max_timeframes: int = 16):
        super().__init__()
        # Staerkere (nicht die uebliche kleine 0.02-Skala) Initialisierung fuer CLS-Token und
        # Timeframe-Embeddings: bei zu kleiner Start-Skala liegen die Attention-Scores aller
        # Timeframes anfangs so nah beieinander, dass die Fusion leicht in einem fast-uniformen
        # Fixpunkt haengen bleibt (wiederholt beobachteter Kollaps, 2026-07-09/10, unabhaengig
        # von Zielgroessen-Definition und Modellgroesse). Ein bewusst groesserer Start-Abstand
        # zwischen den Timeframe-"Identitaeten" erzwingt den Symmetriebruch von Anfang an,
        # statt auf einen guenstigen Zufalls-Seed waehrend des Trainings zu hoffen.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.2)
        self.timeframe_embedding = nn.Parameter(torch.randn(1, max_timeframes, d_model) * 0.2)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        nn.init.xavier_uniform_(self.attention.in_proj_weight, gain=2.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, timeframe_contexts: torch.Tensor):
        """timeframe_contexts: (batch, n_timeframes, d_model) -> (fused (batch, d_model), weights (batch, n_timeframes))"""
        batch_size, n_tf, _ = timeframe_contexts.shape
        x = timeframe_contexts + self.timeframe_embedding[:, :n_tf]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        attn_out, attn_weights = self.attention(x, x, x, need_weights=True, average_attn_weights=True)
        fused = self.norm(attn_out[:, 0] + x[:, 0])
        timeframe_weights = attn_weights[:, 0, 1:]  # Gewicht des CLS-Tokens auf jeden Timeframe
        return fused, timeframe_weights


class StepwiseDecoder(nn.Module):
    """Sagt die Zielvariablen sequentiell voraus (Reihenfolge = TARGET_NAMES): erst die 5

    geometrischen Groessen trend -> range -> close_position -> upper_wick -> lower_wick
    (aus denen reconstruct.py Preis-Koordinaten baut), danach die 3 zusaetzlichen
    Markt-Struktur-Groessen gap_yn -> inside_outside_day -> high_first.

    Jeder Schritt bekommt den fusionierten Kontext plus die Embeddings aller vorherigen
    Schritte (Ground-Truth beim Training / vorhergesagt bei Greedy-Inferenz) als Eingabe --
    analog zum autoregressiven Decoding in Sprachmodellen (Spezifikation Punkt 7: erst Trend,
    dann Volatilitaet, dann Kerzenform).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.step_names = list(TARGET_NAMES)
        self.cardinalities = [TARGET_CARDINALITIES[name] for name in self.step_names]

        self.step_embeddings = nn.ModuleList([nn.Embedding(card, d_model) for card in self.cardinalities])
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, card))
            for card in self.cardinalities
        ])
        # trend (Schritt 0) ist der einzige Decoder-Schritt ganz ohne Vorwissen aus vorherigen
        # Schritten -- am anfaelligsten dafuer, frueh nur den konstanten Trainings-Klassenprior
        # zu lernen (mehrfach beobachtet, 2026-07-09/10). Staerkere Init der letzten Schicht
        # macht die Anfangs-Logits weniger uniform, aehnlich zur Fusion-Attention oben.
        nn.init.xavier_uniform_(self.heads[0][-1].weight, gain=2.0)

    def forward(self, fused_context: torch.Tensor, targets: dict = None) -> dict:
        """
        fused_context: (batch, d_model)
        targets: dict {name: LongTensor(batch,)} fuer Teacher Forcing, oder None fuer Greedy-Decoding.
        Gibt dict {name: logits (batch, card)} zurueck.
        """
        logits = {}
        running_context = fused_context
        for i, name in enumerate(self.step_names):
            step_logits = self.heads[i](running_context)
            logits[name] = step_logits
            prev_class = targets[name] if targets is not None else step_logits.argmax(dim=-1)
            running_context = running_context + self.step_embeddings[i](prev_class)
        return logits


class MarketTransformer(nn.Module):
    """Gesamtmodell: pro Timeframe ein eigener Encoder (unterschiedliche Dynamik je Zeitebene),

    Multi-Timeframe-Attention-Fusion, stufenweiser Decoder ueber die Zielvariablen.
    """

    def __init__(self, n_features: int, timeframes: list, window_sizes: dict,
                 d_model: int = 256, nhead: int = 8, num_encoder_layers: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.timeframes = list(timeframes)
        self.timeframe_encoders = nn.ModuleDict({
            tf: TimeframeEncoder(n_features, d_model, nhead, num_encoder_layers, dim_feedforward,
                                  dropout, max_len=window_sizes[tf])
            for tf in self.timeframes
        })
        self.fusion = TimeframeFusion(d_model, nhead, dropout, max_timeframes=max(16, len(self.timeframes)))
        self.decoder = StepwiseDecoder(d_model, dropout)

    def _fuse(self, features_by_timeframe: dict):
        contexts = [self.timeframe_encoders[tf](features_by_timeframe[tf]) for tf in self.timeframes]
        contexts = torch.stack(contexts, dim=1)  # (batch, n_timeframes, d_model)
        return self.fusion(contexts)

    def forward(self, features_by_timeframe: dict, targets: dict = None):
        """Gibt (logits_dict, timeframe_weights) zurueck. `targets` aktiviert Teacher Forcing."""
        fused, timeframe_weights = self._fuse(features_by_timeframe)
        logits = self.decoder(fused, targets)
        return logits, timeframe_weights

    def compute_loss(self, features_by_timeframe: dict, targets: dict, trend_diversity_weight: float = 0.0):
        """`trend_diversity_weight`: belohnt hohe STANDARDABWEICHUNG der Einzelvorhersagen

        P(bullish) ueber das Batch -- nicht die Entropie des Batch-DURCHSCHNITTS (erste Version
        dieses Regularisierers, verworfen 2026-07-10: ein Modell, das JEDES Beispiel gleich
        knapp Richtung bullish kippt -- z.B. durchgehend P(bullish)=0.565 -- hat eine fast
        maximale Durchschnitts-Entropie [0.435,0.565] UND trotzdem eine komplett kollabierte
        argmax-Entscheidung, weil 0.565 > 0.435 fuer jedes einzelne Beispiel gilt. Durchschnitts-
        Ausgeglichenheit und echte Beispiel-zu-Beispiel-Differenzierung sind zwei verschiedene
        Dinge -- der urspruengliche Regularisierer bestrafte die falsche Groesse. Eine niedrige
        Streuung (alle Vorhersagen nah beieinander, wie im beobachteten Fall) wird jetzt direkt
        bestraft, unabhaengig davon wie ausgeglichen der Durchschnitt zufaellig aussieht.
        """
        logits, timeframe_weights = self.forward(features_by_timeframe, targets=targets)
        losses = {name: F.cross_entropy(logits[name], targets[name]) for name in logits}
        total = sum(losses.values())
        if trend_diversity_weight > 0:
            bullish_probs = F.softmax(logits['trend'], dim=-1)[:, 1]
            spread = torch.std(bullish_probs, unbiased=False)
            total = total - trend_diversity_weight * spread
        return total, losses, timeframe_weights

    @torch.no_grad()
    def predict_beam(self, features_by_timeframe: dict, beam_width: int = 5) -> dict:
        """Beam Search ueber die kategorialen Decoding-Schritte (Spezifikation Punkt 11).

        Erwartet Batch-Groesse 1 (Beam Search wird pro Beispiel einzeln durchgefuehrt).
        """
        self.eval()
        fused, timeframe_weights = self._fuse(features_by_timeframe)
        device = fused.device

        # Beam-Eintrag: (laufender Kontext, gewaehlte Klassen, Log-Wahrscheinlichkeit JEDES
        # einzelnen Schritts, kumulierte Log-Wahrscheinlichkeit). Die einzelnen Schritt-
        # Wahrscheinlichkeiten werden separat mitgefuehrt (nicht nur die Summe), damit man
        # hinterher z.B. "wie sicher war sich das Modell beim Trend" beantworten kann --
        # wichtig fuer eine Konfidenz-Schwelle vor der Trade-Ausfuehrung.
        beams = [(fused, [], [], 0.0)]
        for i, name in enumerate(self.decoder.step_names):
            candidates = []
            for running_context, chosen, step_log_probs, cum_log_prob in beams:
                step_logits = self.decoder.heads[i](running_context)
                log_probs = F.log_softmax(step_logits, dim=-1).squeeze(0)
                k = min(beam_width, log_probs.size(0))
                topk = torch.topk(log_probs, k)
                for log_p, class_id in zip(topk.values.tolist(), topk.indices.tolist()):
                    class_tensor = torch.tensor([class_id], device=device)
                    next_context = running_context + self.decoder.step_embeddings[i](class_tensor)
                    candidates.append((next_context, chosen + [class_id], step_log_probs + [log_p], cum_log_prob + log_p))
            candidates.sort(key=lambda b: b[3], reverse=True)
            beams = candidates[:beam_width]

        _, best_classes, best_step_log_probs, best_log_prob = beams[0]
        result = dict(zip(self.decoder.step_names, best_classes))
        result['log_prob'] = best_log_prob
        result['step_probabilities'] = {
            name: math.exp(lp) for name, lp in zip(self.decoder.step_names, best_step_log_probs)
        }
        result['timeframe_weights'] = dict(zip(self.timeframes, timeframe_weights.squeeze(0).tolist()))
        return result
