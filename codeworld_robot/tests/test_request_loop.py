import json
from types import SimpleNamespace
from codeworld_robot.io import atomic
from codeworld_robot.request_loop import serve_requests


def test_queue_records_errors_and_stops_without_reexecuting_receipts(tmp_path):
    io = tmp_path/'robot_io'
    atomic(io/'requests/a.json', {'op': 'already_done'})
    atomic(io/'responses/a.json', {'status': 'EXECUTED'})
    atomic(io/'requests/b.json', {'op': 'invalid'})
    atomic(io/'requests/c.json', {'op': 'finish'})
    atomic(io/'requests/d.json', {'op': 'must_not_run'})
    import os
    for index, name in enumerate('abcd'):
        os.utime(io/f'requests/{name}.json', ns=(index+1, index+1))
    seen = []; stopped = False
    def dispatch(request):
        nonlocal stopped
        seen.append(request['op'])
        if request['op'] == 'invalid':raise ValueError('bad request')
        stopped = True
        return {'status': 'CLOSED'}
    timeline = SimpleNamespace(operation=None)
    serve_requests(SimpleNamespace(is_running=lambda: True), tmp_path, dispatch, timeline,
                   should_stop=lambda: stopped, steps=lambda: 0)
    assert seen == ['invalid', 'finish']
    error = json.loads((io/'responses/b.json').read_text())
    assert error['status'] == 'ERROR' and error['request_id'] == 'b'
    assert 'bad request' in error['error']
    assert not (io/'responses/d.json').exists()
