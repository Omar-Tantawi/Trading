"""Server-side SVG charts (spec section 5)."""
import xml.etree.ElementTree as ET

from dashboard.svg import calibration_svg, skill_svg

NS = "{http://www.w3.org/2000/svg}"


def _parse(svg):
    return ET.fromstring(svg)


def test_calibration_dots_and_diagonal():
    bins = [{"lo": 0.0, "hi": 0.1, "n": 100, "mean_p": 0.05, "freq": 0.04},
            {"lo": 0.1, "hi": 0.2, "n": 0, "mean_p": None, "freq": None},
            {"lo": 0.6, "hi": 0.7, "n": 50, "mean_p": 0.65, "freq": 0.70}]
    root = _parse(calibration_svg(bins, 0.012, "xgb_v1 <b>&"))
    assert len(root.findall(f".//{NS}circle")) == 2
    assert any(l.get("stroke-dasharray") for l in root.findall(f".//{NS}line"))
    text = "".join(t.text or "" for t in root.iter(f"{NS}text"))
    assert "xgb_v1 <b>&" in text and "ECE 0.012" in text


def test_skill_bars_and_negative_below_zero():
    folds = [{"fold": 0, "test_start": "2020-01-01T01:00:00+00:00", "n": 10, "skill": 0.01},
             {"fold": 1, "test_start": "2020-07-01T01:00:00+00:00", "n": 10, "skill": -0.02}]
    root = _parse(skill_svg(folds, 0.005, 0.001, 0.009, "logreg"))
    rects = root.findall(f".//{NS}rect[@class='bar']")
    assert len(rects) == 2
    zero = float(root.find(f".//{NS}line[@class='zero']").get("y1"))
    assert float(rects[1].get("y")) >= zero - 1e-9
    assert float(rects[0].get("y")) + float(rects[0].get("height")) <= zero + 1e-9
    text = " ".join(t.text or "" for t in root.iter(f"{NS}text"))
    assert "2020-07" in text and "+0.5%" in text


def test_empty_inputs_say_no_data():
    assert "no data" in calibration_svg([], None, "x")
    assert "no data" in skill_svg([], 0.0, None, None, "x")
    _parse(calibration_svg([], None, "x")); _parse(skill_svg([], 0.0, None, None, "x"))
