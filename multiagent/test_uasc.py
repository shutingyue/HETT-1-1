import torch

from multiagent.models.uasc import UASCController, build_uasc_labels


def test_uasc_controller_smoke():
    torch.manual_seed(0)
    controller = UASCController(hidden_size=768)
    coarse_feat = torch.randn(2, 768)
    fine_feat = torch.randn(2, 768)
    progress = torch.rand(2, 1)
    pred_target = torch.tensor([[1.0, 2.0], [30.0, 40.0]])
    current_pos = torch.tensor([[1.0, 1.0], [10.0, 12.0]])
    goal_pos = torch.tensor([[2.0, 2.0], [25.0, 38.0]])
    labels = build_uasc_labels(pred_target, current_pos, goal_pos, progress_gt=progress)

    output = controller(
        coarse_feat=coarse_feat,
        fine_feat=fine_feat,
        progress=progress,
        pred_target=pred_target,
        current_pos=current_pos,
        goal_pos=goal_pos,
        labels=labels,
        return_debug=True,
    )
    assert output["coarse_conf"].shape == (2, 1)
    assert output["stage_prob"].shape == (2, 1)
    assert output["stop_prob"].shape == (2, 1)
    assert "uasc_total" in output["losses"]
    output["losses"]["uasc_total"].backward()


def test_uasc_optional_inputs_and_ignore_labels():
    controller = UASCController(hidden_size=768)
    none_output = controller()
    assert none_output["coarse_conf"].shape == (1, 1)
    assert none_output["losses"] == {}

    ignored_labels = {
        "conf_label": torch.full((2, 1), -100.0),
        "stage_label": torch.full((2, 1), -100.0),
        "stop_label": torch.full((2, 1), -100.0),
    }
    output = controller(
        coarse_feat=torch.randn(2, 3, 768),
        fine_feat=None,
        labels=ignored_labels,
    )
    assert torch.isfinite(output["losses"]["uasc_total"])
    output["losses"]["uasc_total"].backward()


if __name__ == "__main__":
    test_uasc_controller_smoke()
    test_uasc_optional_inputs_and_ignore_labels()
    print("UASC smoke test passed.")
