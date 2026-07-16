from rad_benchmark.vlm import render_response_mask


def test_normalized_box_to_mask() -> None:
    response = {
        "is_anomalous": True,
        "coordinate_space": "normalized",
        "boxes": [{"bbox": [0.25, 0.25, 0.75, 0.75]}],
    }
    mask = render_response_mask(response, 8, 8)
    assert mask.getpixel((3, 3)) == 255
    assert mask.getpixel((0, 0)) == 0
