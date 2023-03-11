from Analyse import *
from Image import *



# ------Tests-----#

print(f"#\n----------------Image----------------\n#")
I1 = Img('Base/img/poissons_falsifies.jpg')
I2 = Img('Base/img/poissons_falsifies.jpg')
#I.show()
# print(I.tableau)
I1.drawSift('Base/img/siftC.jpg', False)
I2.drawSift('Base/img/siftG.jpg', True)
#print(I.G())
