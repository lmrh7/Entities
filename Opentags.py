import pickle
objs = []
#

# f = open(''mon ami gabi_revTags', 'rb')
f = open('fileName', 'rb')
while 1:
    try:
        objs.append(pickle.load(f))
    except EOFError:
        break

for object in objs:
    print object