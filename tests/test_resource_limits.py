import pytest

from core.limits import BusyError, WorkLimiter


def test_global_concurrency_and_exception_release():
    limiter = WorkLimiter(concurrency=1)
    with pytest.raises(RuntimeError):
        with limiter.render("alice", "shared-ip"):
            with pytest.raises(BusyError, match="slots"):
                with limiter.render("bob", "shared-ip"):
                    pytest.fail("must not run")
            raise RuntimeError("training failed")
    with limiter.render("bob", "shared-ip"):
        assert limiter.active == 1
    assert limiter.active == 0


def test_training_cooldown_refills_and_clients_are_independent():
    now = [0.0]
    limiter = WorkLimiter(clock=lambda: now[0])
    limiter.training("alice", "campus")
    limiter.training("alice", "campus")
    with pytest.raises(BusyError) as error:
        limiter.training("alice", "campus")
    assert error.value.retry_after == 5
    limiter.training("bob", "campus")
    now[0] = 5
    limiter.training("alice", "campus")


def test_new_sessions_cannot_bypass_ip_budget():
    limiter = WorkLimiter(clock=lambda: 0)
    for i in range(30):
        limiter.training(str(i), "same-ip")
    with pytest.raises(BusyError):
        limiter.training("new-session", "same-ip")
    limiter.training("different-user", "different-ip")


def test_tracking_memory_is_bounded_and_expires():
    now = [0]
    limiter = WorkLimiter(max_clients=2, clock=lambda: now[0])
    limiter.training("one", "ip-one")
    with pytest.raises(BusyError):
        limiter.training("two", "ip-two")
    assert len(limiter.buckets) == 2
    now[0] = 3601
    limiter.training("two", "ip-two")
    assert len(limiter.buckets) == 2
