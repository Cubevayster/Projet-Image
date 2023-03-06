from Analyse import *
from Image import *

#------Tests-----#

print(f"#\n----------------Image----------------\n#")
I = Img('../img/cristalline2.jpg')
#I.show()
#print(I.tableau)
print(I.G())
print(f"#\n----------------Timer---------------\n#")
timer = Timer()
timer.start()

for i in range(0, 10):
  i = i**2
timer.end('s')

timer2 = Timer()
c = []
timer2.acces_memmoire("x,y","x.append(y)",3,3)

class c:
  def __init__(self):
    self.x = 0
    
  def up(n):
    for i in range(0,n) : i=i

C = c()
timer2.start()
c.up(1000000)
timer2.end()
timer.complexite(C)

print('------------')
timer3 = Timer()
timer3.data_arbre(3,3)
#---------------#
