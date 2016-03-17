import pickle
objs = []
#f = open('mon_ami_revTags', 'rb')
f = open('mon_ami_capTags', 'rb')
while 1:
    try:
        objs.append(pickle.load(f))
    except EOFError:
        break

for object in objs:
    print object