import glob

import bionetgen as bng

models = glob.glob("*.bngl")
total = len(models)
succ = []
fail = []
success = 0
fails = 0
for model in models:
    try:
        m = bng.bngmodel(model)
        success += 1
        succ.append(model)
        # mname = model.replace(".bngl","")
        # with open("{}_loaded.bngl".format(mname), 'w') as f:
        #     f.write(str(m))
    except:
        print(f"can't do model {model}")
        fails += 1
        fail.append(model)

print(f"succ: {success}")
print(sorted(succ))
print(f"fail: {fails}")
print(sorted(fail))
