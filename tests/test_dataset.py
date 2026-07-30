import pytest
from src.dataset.structures import Point, Stroke, Trajectory
from src.dataset.tokenizer import CoordinateTokenizer
from src.dataset.parser import IIITHWParser

def test_structures():
    pt = Point(x=10.0, y=20.0, p=1)
    assert pt.x == 10.0
    
    stroke = Stroke(points=[pt, Point(x=11.0, y=21.0, p=0)])
    assert len(stroke.points) == 2
    
    traj = Trajectory(strokes=[stroke], text="hello")
    arr = traj.to_array()
    assert len(arr) == 2
    assert arr[0] == [10.0, 20.0, 1]

def test_tokenizer():
    tokenizer = CoordinateTokenizer(grid_size=256)
    
    # Check quantization bounds
    assert tokenizer.quantize(0.0, 0.0, 100.0) == tokenizer.coord_offset
    assert tokenizer.quantize(100.0, 0.0, 100.0) == tokenizer.coord_offset + 255
    
    # Tokenize dummy trajectory
    pt1 = Point(x=0.0, y=0.0, p=1)
    pt2 = Point(x=100.0, y=100.0, p=0)
    stroke = Stroke(points=[pt1, pt2])
    traj = Trajectory(strokes=[stroke])
    
    tokens = tokenizer.tokenize_trajectory(traj)
    # Expected: [X1, Y1, X2, Y2, PEN_LIFT, EOS]
    # X1, Y1 should map to min (coord_offset)
    # X2, Y2 should map to max (coord_offset + 255)
    assert tokens[0] == tokenizer.coord_offset
    assert tokens[1] == tokenizer.coord_offset
    assert tokens[2] == tokenizer.coord_offset + 255
    assert tokens[3] == tokenizer.coord_offset + 255
    assert tokens[4] == tokenizer.pen_lift_token
    assert tokens[5] == tokenizer.eos_token
