from freetype import *
import numpy as np
face = Face('env_test/arial.ttf')
flags = FT_LOAD_DEFAULT | FT_LOAD_NO_BITMAP
face.set_char_size(10*10)
face.load_char('E', flags )
slot = face.glyph
outline = slot.outline
points = np.asarray(face.glyph.outline.points)
points[:,0] = points[:,0] - 0
points[:,1] = points[:,1] - 0
print(points)