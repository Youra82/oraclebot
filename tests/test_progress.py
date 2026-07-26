import io

from oraclebot.utils import progress


class _FakeTTYStream(io.StringIO):
    def isatty(self):
        return True


def test_render_progress_writes_nothing_when_not_a_tty(monkeypatch, capsys):
    monkeypatch.setattr(progress.sys.stdout, 'isatty', lambda: False)
    progress.render_progress('Test', 1, 10, progress.time.time())
    progress.finish_progress()
    captured = capsys.readouterr()
    assert captured.out == ''


def test_render_progress_writes_a_bar_when_tty():
    fake_stdout = _FakeTTYStream()
    original_stdout = progress.sys.stdout
    progress.sys.stdout = fake_stdout
    try:
        progress.render_progress('Test', 5, 10, progress.time.time())
        progress.finish_progress()
    finally:
        progress.sys.stdout = original_stdout

    output = fake_stdout.getvalue()
    assert '\r' in output
    assert 'Test: 5/10 (50%)' in output
    assert output.endswith('\n')


def test_render_progress_caps_percentage_at_100():
    fake_stdout = _FakeTTYStream()
    original_stdout = progress.sys.stdout
    progress.sys.stdout = fake_stdout
    try:
        progress.render_progress('Test', 15, 10, progress.time.time())
    finally:
        progress.sys.stdout = original_stdout

    assert '(100%)' in fake_stdout.getvalue()
