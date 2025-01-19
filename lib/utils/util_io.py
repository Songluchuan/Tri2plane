
class OBJ():
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.ft = []
        self.vt = []


def load_obj(path, attribute='all'):
    obj = OBJ()
    with open(path, 'r', encoding = "utf-8") as f:
        while(True):
            line = f.readline().strip(' ')
            if not line:
                break
    
            if line.startswith('v ') and (attribute == 'v' or attribute == 'all'):
                line = line.split()
                obj.vertices.append([float(line[1]), float(line[2]), float(line[3])])
            elif line.startswith('vt ') and (attribute == 'vt' or attribute == 'all'):
                line = line.split()
                obj.vt.append([float(line[1]), float(line[2])])
            elif line.startswith('f ') and (attribute == 'f' or attribute == 'all'):
                line = line.split()[1:]
                obj.faces.append([int(term.split('/')[0]) - 1 for term in line])
                obj.ft.append([int(term.split('/')[1]) - 1 for term in line])

    return obj


def write_obj(path, obj):
    with open(path, 'w', encoding = "utf-8") as f:
        for i in range(len(obj.vertices)):
            term = obj.vertices[i]
            print('v %f %f %f' % (term[0], term[1], term[2]), file=f)
        if obj.vt:
            for i in range(len(obj.vt)):
                term = obj.vt[i]
                print('vt %f %f' % (term[0], term[1]), file=f)
        print('s off', file=f)
        for i in range(len(obj.faces)):
            term_f = obj.faces[i] + 1
            if obj.ft:
                term_ft = obj.ft[i] + 1
                if len(term_f) == 4:
                    print('f %d/%d %d/%d %d/%d %d/%d' % (term_f[0], term_ft[0], 
                                                        term_f[1], term_ft[1], 
                                                        term_f[2], term_ft[2], 
                                                        term_f[3], term_ft[3]), file=f)
                elif len(term_f) == 3:
                    print('f %d/%d %d/%d %d/%d' % (term_f[0], term_ft[0], 
                                                term_f[1], term_ft[1], 
                                                term_f[2], term_ft[2]), file=f)
            else:
                if len(term_f) == 4:
                    print('f %d %d %d %d' % (term_f[0], term_f[1], term_f[2], term_f[3]), file=f)
                elif len(term_f) == 3:
                    print('f %d %d %d' % (term_f[0], term_f[1], term_f[2]), file=f)
    return None


