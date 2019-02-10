from matplotlib import pyplot
import numpy, weakref
a = numpy.arange(int(1e3))
fig = pyplot.Figure()
ax  = fig.add_subplot(111)
lines = ax.plot(a)

#l = lines.pop(0)
#wl = weakref.ref(l)  # create a weak reference to see if references still exist
##                      to this object
#print wl  # not dead
#l.remove()
#print wl  # not dead
#del l
#print wl  # dead  (remove eit

pyplot.imshow(pyplot.imread('C:/Users/aashi/Downloads/cat.jpg'))
h1= pyplot.plot([1 ,50],[1, 50],Color = 'b');
pyplot.draw()
