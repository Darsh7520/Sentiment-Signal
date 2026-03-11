from qiskit import QuantumCircuit
from qiskit_aer import Aer
import numpy as np
from PIL import Image

#QRNG
def qrng(w,h):
    qc = QuantumCircuit(w, w)
    qc.h(range(w))
    qc.measure(range(w), range(w))

    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(qc, shots = h, memory=True)
    result = job.result()

    bits = result.get_memory(qc)

    RandMat1 = np.array([[int(b) for b in s] for s in bits], dtype=np.uint8)
    #print(RandMat1.shape)
    return RandMat1

#Generate shares
def generate_shares(RandMat1, image_path):
    image = Image.open(image_path).convert('1')
    image_arr = np.array(image, dtype=np.uint8)
    #print("image: ", image_arr)
    RandMat2 = np.array(np.logical_xor(image_arr, RandMat1),dtype=np.uint8)

    share_1 = Image.fromarray(RandMat1*255)
    share_2 = Image.fromarray(RandMat2*255)
    share_1.save("share_1.png")
    share_2.save("share_2.png")
    return share_1,share_2



def combine_shares(RandMat1,RandMat2):
    share_1 = Image.open("share_1.png")
    share_2 = Image.open("share_2.png")
    share1_arr = np.array(share_1)
    share2_arr = np.array(share_2)
    rec_arr = np.logical_xor(share1_arr,share2_arr)
    rec_img = Image.fromarray(rec_arr.astype('uint8')*255)
    rec_img.save("Recovered.png")
    return rec_arr

def shannon_entropy(bits):
    values, counts = np.unique(bits, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return entropy



def main():
    image_path = input('Enter file name:')
    image = Image.open(image_path)
    w,h = image.size
    randq = qrng(w,h)
    ent = shannon_entropy(randq)
    randc = np.random.randint(0, w, size=randq.shape,dtype = np.uint8)
    entc = shannon_entropy(randc)
    print(f"Quantum generated matrix entropy: {ent}")
    print(f"Classically generated matrix entropy: {ent}" )
    share1,share2 = generate_shares(randq,image_path)
    rec = combine_shares(share1,share2)
    


if __name__ == "__main__":
    main() 