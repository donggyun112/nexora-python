from uuid import UUID

import semora


def test_new_run_id_is_a_full_uuid7_with_the_requested_prefix() -> None:
    run_id = semora.new_run_id("ui")

    assert run_id.startswith("ui-")
    encoded = run_id.removeprefix("ui-")
    value = UUID(encoded)
    assert value.version == 7
    assert encoded == value.hex


def test_new_run_ids_sort_in_creation_order() -> None:
    run_ids = [semora.new_run_id() for _ in range(100)]

    assert run_ids == sorted(run_ids)
