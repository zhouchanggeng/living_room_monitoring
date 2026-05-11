import unittest

from runtime.follow_controller import (
    FollowConfig,
    FollowController,
    FollowMode,
    FollowState,
    TrackedObject,
)


def obj(track_id, cls_name="Adult", bbox=(260, 140, 380, 340), conf=0.9):
    return TrackedObject(
        track_id=track_id,
        class_name=cls_name,
        bbox_xyxy=bbox,
        confidence=conf,
    )


class FollowControllerTest(unittest.TestCase):
    def test_stays_idle_when_no_target_has_ever_been_locked(self):
        controller = FollowController(FollowConfig(follow_mode=FollowMode.AUTO))

        command = controller.update([], (640, 384))

        self.assertEqual(command.state, FollowState.IDLE)
        self.assertEqual(command.pan_speed, 0.0)
        self.assertEqual(command.tilt_speed, 0.0)

    def test_keeps_locked_track_when_multiple_targets_are_visible(self):
        controller = FollowController(FollowConfig(follow_mode=FollowMode.AUTO))

        first = controller.update([obj(1), obj(2, bbox=(100, 120, 220, 320))], (640, 384))
        second = controller.update([obj(1), obj(2, bbox=(300, 120, 420, 320))], (640, 384))

        self.assertEqual(first.target_track_id, 1)
        self.assertEqual(second.target_track_id, 1)
        self.assertEqual(second.state, FollowState.TRACK)

    def test_filters_targets_by_follow_mode(self):
        controller = FollowController(FollowConfig(follow_mode=FollowMode.CHILD))

        command = controller.update(
            [
                obj(1, cls_name="Adult", bbox=(280, 100, 420, 360), conf=0.99),
                obj(2, cls_name="Kid", bbox=(260, 130, 360, 310), conf=0.8),
            ],
            (640, 384),
        )

        self.assertEqual(command.target_track_id, 2)
        self.assertEqual(command.target_class, "Kid")

    def test_short_loss_holds_last_target_then_searches_after_timeout(self):
        controller = FollowController(
            FollowConfig(
                follow_mode=FollowMode.ADULT,
                fps=10.0,
                hold_seconds=0.2,
                lost_seconds=0.4,
            )
        )

        controller.update([obj(7)], (640, 384))
        hold = controller.update([], (640, 384))
        search = None
        for _ in range(5):
            search = controller.update([], (640, 384))

        self.assertEqual(hold.state, FollowState.HOLD)
        self.assertEqual(hold.target_track_id, 7)
        self.assertEqual(search.state, FollowState.SEARCH)
        self.assertIsNone(search.target_track_id)
        self.assertNotEqual(search.pan_speed, 0.0)

    def test_dead_zone_suppresses_small_pan_tilt_commands(self):
        controller = FollowController(
            FollowConfig(
                follow_mode=FollowMode.ADULT,
                dead_zone_x_ratio=0.03,
                dead_zone_y_ratio=0.03,
            )
        )

        command = controller.update([obj(3, bbox=(300, 170, 340, 214))], (640, 384))

        self.assertEqual(command.state, FollowState.TRACK)
        self.assertEqual(command.pan_speed, 0.0)
        self.assertEqual(command.tilt_speed, 0.0)

    def test_large_error_generates_limited_speed_command(self):
        controller = FollowController(
            FollowConfig(
                follow_mode=FollowMode.ADULT,
                kp_x=0.01,
                kp_y=0.01,
                max_pan_speed=1.0,
                max_tilt_speed=0.8,
            )
        )

        command = controller.update([obj(5, bbox=(520, 300, 620, 380))], (640, 384))

        self.assertEqual(command.pan_speed, 1.0)
        self.assertEqual(command.tilt_speed, 0.8)

    def test_returns_simulation_direction_and_coordinate_delta(self):
        controller = FollowController(
            FollowConfig(
                follow_mode=FollowMode.ADULT,
                kp_x=0.01,
                kp_y=0.01,
                max_pan_speed=1.0,
                max_tilt_speed=1.0,
            )
        )

        command = controller.update([obj(8, bbox=(420, 250, 520, 350))], (640, 400))

        self.assertEqual(command.direction_x, "right")
        self.assertEqual(command.direction_y, "down")
        self.assertAlmostEqual(command.normalized_error_xy[0], 0.46875)
        self.assertAlmostEqual(command.normalized_error_xy[1], 0.5)
        self.assertEqual(command.sim_delta_xy, (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
