# API 包

## aigen.py

### create_beam()

```python
Vertex = np.typing.NDArray[np.floating]          # 形状 (3,) 的浮点向量
Triangle = Tuple[Vertex, Vertex, Vertex]         # 三角形的三个顶点
BeamFaces = List[Triangle]                       # 所有三角形组成的列表

def create_beam(
    start: Union[Sequence[float], np.ndarray],
    end:   Union[Sequence[float], np.ndarray],
    half_width: float = 0.05
) -> BeamFaces:
```

为给定中心点创建立方体的三角形面

### create_cube()
``` python
Vertex   = Tuple[float, float, float]          # 立方体的一个顶点 (x, y, z)
Triangle = Tuple[Vertex, Vertex, Vertex]       # 三角形由三个顶点组成
CubeFaces = List[Triangle]                     # 所有三角形面的列表

def create_cube(
    center: Sequence[float],
    half_size: float = 0.05
) -> CubeFaces:
```

为给定中心点创建立方体的三角形面。


