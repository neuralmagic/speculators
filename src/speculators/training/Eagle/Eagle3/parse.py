import numpy as np
file="output.txt"
text=open(file).read()
# print(text)


lines=text.split("SpecDecoding metrics:")
tokens=[]
acceptances=[]
print(len(lines))
for line in lines[1:]:

    data=(line.split("\n")[0])
    drafted=eval((data.split("Drafted: ")[1]).split(" ")[0])
    tokens.append(drafted)
    accepted=data.split("Per-position acceptance rate: ")[1]

    accepted=accepted.split(" ")

    accepted=[eval(x.strip(",")) for x in accepted]
    accepted=np.array(accepted)
    acceptances.append(accepted)
acceptances=np.array(acceptances)
tokens=np.array(tokens)

weighted=np.sum(acceptances*tokens[:, None], axis=0)/np.sum(tokens)

print(np.round(weighted, decimals=3))


weighted=np.concat((np.array([1]), weighted))
print("conditional")
print(np.round(weighted[1:]/weighted[:-1], decimals=3),)