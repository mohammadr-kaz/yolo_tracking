import math


class CameraConfig:
    """Camera field-of-view settings and azimuth/elevation calculation."""

    def __init__(self, horizontal_fov=60.0, vertical_fov=45.0):
        """
        Args:
            horizontal_fov: camera horizontal field of view (degrees)
            vertical_fov: camera vertical field of view (degrees)
        """
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov

    def calculate_angles(self, target_x, target_y, frame_width, frame_height):
        """
        Compute azimuth/elevation of a target relative to the frame center.

        Args:
            target_x: target center, horizontal pixel position
            target_y: target center, vertical pixel position
            frame_width: frame width in pixels
            frame_height: frame height in pixels

        Returns:
            (azimuth, elevation) in degrees.
            azimuth: positive = right, negative = left
            elevation: positive = up, negative = down
        """
        center_x = frame_width / 2
        center_y = frame_height / 2

        normalized_x = (target_x - center_x) / (frame_width / 2)
        normalized_y = (center_y - target_y) / (frame_height / 2)

        max_azimuth_rad = math.radians(self.horizontal_fov / 2)
        azimuth = math.degrees(normalized_x * max_azimuth_rad)

        max_elevation_rad = math.radians(self.vertical_fov / 2)
        elevation = math.degrees(normalized_y * max_elevation_rad)

        return azimuth, elevation

    def set_fov(self, horizontal_fov, vertical_fov):
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
