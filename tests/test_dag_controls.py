"""One control, one cause, for the task-graph playground.

Each control maps to one field of ``Scenario``, and each field reaches exactly
one place: the failing task and its failure count reach only the handlers, the
attempt count reaches only ``run``, the two edge switches reach only the edges.
"""

import dataclasses

from dag_playground.scheduler import ALWAYS, PIPELINE_EDGES, Scenario


def test_the_scenario_has_one_field_per_control():
    fields = [field.name for field in dataclasses.fields(Scenario)]

    assert fields == ["failing_task", "failures", "attempts", "guard", "cycle"]


def test_failure_settings_do_not_touch_the_edges():
    baseline = Scenario().edges()

    for scenario in (
        Scenario(failing_task="очистка"),
        Scenario(failures=ALWAYS),
        Scenario(failures=0),
        Scenario(attempts=1),
    ):
        assert scenario.edges() == baseline


def test_the_edge_switches_do_not_touch_the_handlers():
    baseline = set(Scenario().handlers())

    assert set(Scenario(guard=False).handlers()) == baseline
    assert set(Scenario(cycle=True).handlers()) == baseline


def test_each_edge_switch_changes_exactly_one_edge():
    lecture = set(PIPELINE_EDGES)

    assert len(lecture ^ set(Scenario(guard=False).edges())) == 1
    assert len(lecture ^ set(Scenario(cycle=True).edges())) == 1


def test_no_failure_means_no_handler_at_all():
    assert Scenario(failing_task=None).handlers() == {}
    assert Scenario(failures=0).handlers() == {}


def test_scenarios_are_hashable_for_the_cache():
    assert hash(Scenario()) == hash(Scenario())
