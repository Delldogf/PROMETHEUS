"""
Geometry Engine: Geometric reasoning and computation.

This engine handles:
- Points, lines, circles, triangles
- Distances and angles
- Transformations
- Coordinate geometry
- Verification of geometric properties

WHAT THIS FILE DOES:
- Represents geometric objects
- Computes geometric properties
- Verifies geometric conditions
- Supports both synthetic and coordinate approaches
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
import math


@dataclass
class Point:
    """A point in 2D space."""
    x: float
    y: float
    name: Optional[str] = None
    
    def distance_to(self, other: Point) -> float:
        """Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def midpoint(self, other: Point) -> Point:
        """Midpoint between this and another point."""
        return Point((self.x + other.x) / 2, (self.y + other.y) / 2)
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.name}({self.x}, {self.y})"
        return f"({self.x}, {self.y})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9


@dataclass
class Line:
    """
    A line in 2D space.
    
    Represented in general form: ax + by + c = 0
    """
    a: float
    b: float
    c: float
    
    @classmethod
    def from_two_points(cls, p1: Point, p2: Point) -> Line:
        """Create a line through two points."""
        a = p2.y - p1.y
        b = p1.x - p2.x
        c = p2.x * p1.y - p1.x * p2.y
        return cls(a, b, c)
    
    @classmethod
    def from_point_slope(cls, p: Point, slope: float) -> Line:
        """Create a line through a point with given slope."""
        # y - p.y = slope * (x - p.x)
        # slope*x - y + (p.y - slope*p.x) = 0
        a = slope
        b = -1
        c = p.y - slope * p.x
        return cls(a, b, c)
    
    def contains_point(self, p: Point, tolerance: float = 1e-9) -> bool:
        """Check if a point lies on this line."""
        return abs(self.a * p.x + self.b * p.y + self.c) < tolerance
    
    def is_parallel_to(self, other: Line, tolerance: float = 1e-9) -> bool:
        """Check if this line is parallel to another."""
        return abs(self.a * other.b - self.b * other.a) < tolerance
    
    def intersection(self, other: Line) -> Optional[Point]:
        """Find intersection point with another line."""
        det = self.a * other.b - self.b * other.a
        if abs(det) < 1e-9:
            return None  # Parallel lines
        x = (self.b * other.c - other.b * self.c) / det
        y = (other.a * self.c - self.a * other.c) / det
        return Point(x, y)
    
    def perpendicular_through(self, p: Point) -> Line:
        """Create a perpendicular line through a point."""
        # Perpendicular direction: (b, -a)
        return Line(self.b, -self.a, -self.b * p.x + self.a * p.y)
    
    def distance_to_point(self, p: Point) -> float:
        """Distance from a point to this line."""
        return abs(self.a * p.x + self.b * p.y + self.c) / math.sqrt(self.a**2 + self.b**2)
    
    def __str__(self) -> str:
        return f"{self.a}x + {self.b}y + {self.c} = 0"


@dataclass
class Circle:
    """
    A circle in 2D space.
    
    Defined by center and radius.
    """
    center: Point
    radius: float
    
    @classmethod
    def from_three_points(cls, p1: Point, p2: Point, p3: Point) -> Optional[Circle]:
        """Create a circle through three points (circumcircle)."""
        # Use determinant formula
        ax, ay = p1.x, p1.y
        bx, by = p2.x, p2.y
        cx, cy = p3.x, p3.y
        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-9:
            return None  # Collinear points
        
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        
        center = Point(ux, uy)
        radius = center.distance_to(p1)
        return cls(center, radius)
    
    def contains_point(self, p: Point, tolerance: float = 1e-9) -> bool:
        """Check if a point lies on the circle."""
        dist = self.center.distance_to(p)
        return abs(dist - self.radius) < tolerance
    
    def point_inside(self, p: Point) -> bool:
        """Check if a point is inside the circle."""
        return self.center.distance_to(p) < self.radius
    
    def intersection_with_line(self, line: Line) -> List[Point]:
        """Find intersection points with a line."""
        # Distance from center to line
        d = line.distance_to_point(self.center)
        
        if d > self.radius + 1e-9:
            return []  # No intersection
        
        if abs(d - self.radius) < 1e-9:
            # Tangent - one point
            # Project center onto line
            t = -(line.a * self.center.x + line.b * self.center.y + line.c) / (line.a**2 + line.b**2)
            px = self.center.x + t * line.a
            py = self.center.y + t * line.b
            return [Point(px, py)]
        
        # Two intersection points
        # This is simplified - full implementation would need proper projection
        # For now, use substitution method
        return []  # TODO: Implement full intersection
    
    def __str__(self) -> str:
        return f"Circle(center={self.center}, r={self.radius})"


@dataclass
class Triangle:
    """
    A triangle defined by three vertices.
    """
    A: Point
    B: Point
    C: Point
    
    @property
    def sides(self) -> Tuple[float, float, float]:
        """Return the side lengths (a, b, c) opposite to vertices (A, B, C)."""
        a = self.B.distance_to(self.C)
        b = self.C.distance_to(self.A)
        c = self.A.distance_to(self.B)
        return (a, b, c)
    
    @property
    def perimeter(self) -> float:
        """Compute the perimeter."""
        a, b, c = self.sides
        return a + b + c
    
    @property
    def area(self) -> float:
        """Compute the area using Heron's formula."""
        a, b, c = self.sides
        s = (a + b + c) / 2
        return math.sqrt(s * (s - a) * (s - b) * (s - c))
    
    @property
    def centroid(self) -> Point:
        """Compute the centroid (intersection of medians)."""
        x = (self.A.x + self.B.x + self.C.x) / 3
        y = (self.A.y + self.B.y + self.C.y) / 3
        return Point(x, y)
    
    @property
    def circumcircle(self) -> Optional[Circle]:
        """Compute the circumscribed circle."""
        return Circle.from_three_points(self.A, self.B, self.C)
    
    @property
    def incircle_radius(self) -> float:
        """Compute the inradius."""
        return self.area / (self.perimeter / 2)
    
    def angle_at(self, vertex: str) -> float:
        """
        Compute the angle at a vertex (in radians).
        
        vertex: 'A', 'B', or 'C'
        """
        a, b, c = self.sides
        
        if vertex == 'A':
            # Angle at A is opposite to side a
            cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        elif vertex == 'B':
            cos_angle = (a**2 + c**2 - b**2) / (2 * a * c)
        else:  # C
            cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        
        # Clamp for numerical stability
        cos_angle = max(-1, min(1, cos_angle))
        return math.acos(cos_angle)
    
    def is_right(self, tolerance: float = 1e-9) -> bool:
        """Check if this is a right triangle."""
        a, b, c = sorted(self.sides)
        return abs(a**2 + b**2 - c**2) < tolerance
    
    def is_isosceles(self, tolerance: float = 1e-9) -> bool:
        """Check if this is an isosceles triangle."""
        a, b, c = self.sides
        return (abs(a - b) < tolerance or 
                abs(b - c) < tolerance or 
                abs(a - c) < tolerance)
    
    def is_equilateral(self, tolerance: float = 1e-9) -> bool:
        """Check if this is an equilateral triangle."""
        a, b, c = self.sides
        return abs(a - b) < tolerance and abs(b - c) < tolerance


@dataclass
class GeometryResult:
    """Result of a geometric computation."""
    success: bool
    result: Any = None
    method: str = ""
    steps: List[str] = field(default_factory=list)
    error: Optional[str] = None


class GeometryEngine:
    """
    Engine for geometric reasoning.
    
    Key capabilities:
    - Compute distances, angles, areas
    - Check geometric properties (collinear, concurrent, etc.)
    - Perform transformations
    - Verify geometric theorems
    """
    
    def __init__(self):
        pass
    
    # ===== Basic Computations =====
    
    def distance(self, p1: Point, p2: Point) -> GeometryResult:
        """Compute distance between two points."""
        d = p1.distance_to(p2)
        return GeometryResult(
            success=True,
            result=d,
            method="Euclidean distance",
            steps=[f"|{p1} - {p2}| = {d}"]
        )
    
    def angle(self, p1: Point, vertex: Point, p2: Point) -> GeometryResult:
        """
        Compute the angle at vertex, between rays to p1 and p2.
        Returns angle in radians.
        """
        # Vectors from vertex to p1 and p2
        v1 = (p1.x - vertex.x, p1.y - vertex.y)
        v2 = (p2.x - vertex.x, p2.y - vertex.y)
        
        # Dot product and magnitudes
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag1 < 1e-9 or mag2 < 1e-9:
            return GeometryResult(success=False, error="Degenerate angle (zero length vector)")
        
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp
        angle_rad = math.acos(cos_angle)
        
        return GeometryResult(
            success=True,
            result=angle_rad,
            method="dot product formula",
            steps=[f"∠{p1}{vertex}{p2} = {math.degrees(angle_rad):.2f}°"]
        )
    
    def triangle_area(self, p1: Point, p2: Point, p3: Point) -> GeometryResult:
        """Compute area of triangle with vertices p1, p2, p3."""
        tri = Triangle(p1, p2, p3)
        area = tri.area
        return GeometryResult(
            success=True,
            result=area,
            method="Heron's formula",
            steps=[f"Area = {area}"]
        )
    
    # ===== Collinearity and Concurrency =====
    
    def are_collinear(self, *points: Point, tolerance: float = 1e-9) -> GeometryResult:
        """Check if three or more points are collinear."""
        if len(points) < 3:
            return GeometryResult(success=True, result=True, method="< 3 points are trivially collinear")
        
        # Use the first two points to define a line
        p1, p2 = points[0], points[1]
        if p1 == p2:
            return GeometryResult(success=False, error="First two points are identical")
        
        line = Line.from_two_points(p1, p2)
        
        for p in points[2:]:
            if not line.contains_point(p, tolerance):
                return GeometryResult(
                    success=True,
                    result=False,
                    method="line containment test",
                    steps=[f"Point {p} is not on line through {p1} and {p2}"]
                )
        
        return GeometryResult(
            success=True,
            result=True,
            method="all points lie on line"
        )
    
    def are_concurrent(self, *lines: Line, tolerance: float = 1e-9) -> GeometryResult:
        """Check if three or more lines are concurrent (meet at a single point)."""
        if len(lines) < 2:
            return GeometryResult(success=True, result=True, method="< 2 lines are trivially concurrent")
        
        # Find intersection of first two lines
        intersection = lines[0].intersection(lines[1])
        if intersection is None:
            return GeometryResult(
                success=True,
                result=False,
                method="first two lines are parallel"
            )
        
        # Check if all other lines pass through this point
        for line in lines[2:]:
            if not line.contains_point(intersection, tolerance):
                return GeometryResult(
                    success=True,
                    result=False,
                    steps=[f"Line {line} does not pass through {intersection}"]
                )
        
        return GeometryResult(
            success=True,
            result=True,
            method="all lines pass through common point",
            steps=[f"Concurrent at {intersection}"]
        )
    
    # ===== Transformations =====
    
    def reflect_point(self, p: Point, line: Line) -> GeometryResult:
        """Reflect a point across a line."""
        # Find foot of perpendicular
        perp = line.perpendicular_through(p)
        foot = line.intersection(perp)
        
        if foot is None:
            return GeometryResult(success=False, error="Could not find foot of perpendicular")
        
        # Reflection is 2*foot - p
        reflected = Point(2 * foot.x - p.x, 2 * foot.y - p.y)
        
        return GeometryResult(
            success=True,
            result=reflected,
            method="reflection across line"
        )
    
    def rotate_point(self, p: Point, center: Point, angle: float) -> GeometryResult:
        """Rotate a point around a center by given angle (in radians)."""
        # Translate to origin
        dx = p.x - center.x
        dy = p.y - center.y
        
        # Rotate
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        new_dx = dx * cos_a - dy * sin_a
        new_dy = dx * sin_a + dy * cos_a
        
        # Translate back
        rotated = Point(center.x + new_dx, center.y + new_dy)
        
        return GeometryResult(
            success=True,
            result=rotated,
            method=f"rotation by {math.degrees(angle):.2f}° around {center}"
        )
    
    # ===== Verification =====
    
    def verify_pythagorean(self, a: float, b: float, c: float, tolerance: float = 1e-9) -> GeometryResult:
        """Verify if a² + b² = c² (assuming c is hypotenuse)."""
        lhs = a**2 + b**2
        rhs = c**2
        holds = abs(lhs - rhs) < tolerance
        
        return GeometryResult(
            success=True,
            result=holds,
            method="Pythagorean theorem check",
            steps=[f"{a}² + {b}² = {lhs}", f"{c}² = {rhs}", f"Equal: {holds}"]
        )
