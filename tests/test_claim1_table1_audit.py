import importlib.util, pathlib, tempfile, json


def test_table_arithmetic_and_honest_scope():
    p = pathlib.Path(__file__).parents[1] / 'src/claim1_table1_audit.py'
    s = importlib.util.spec_from_file_location('c', p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as d:
        m.main(d)
        x = json.load(open(pathlib.Path(d) / 'summary.json'))
        assert x['absolute_reduction_percentage_points'] == 21.85
        assert x['verdict'] == 'inconclusive'
