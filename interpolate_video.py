class sct:
    def __init__(self):
        pass

import argparse,pathlib,time
from queue import Queue, Empty

parser = argparse.ArgumentParser(description='Video reencoder')

parser.add_argument('--vin',  type=pathlib.Path)
parser.add_argument('--vout', type=pathlib.Path)
parser.add_argument('--maxsizein',  type=int, default=16)
parser.add_argument('--maxsizeout', type=int, default=16)
#parser.add_argument('',type=,default=)
parser.add_argument('--netName',type=str,default='DAIN_slowmotion')
parser.add_argument('--model',type=str,default='./model_weights/best.pth')


#parser.add_argument('--exp',type=int,default=1)
parser.add_argument('--multiplier',type=int,default=1)

args = parser.parse_args()
assert args.vin !=None and args.vout!=None




pin   = str(args.vin)
bufin = Queue(maxsize=args.maxsizein)
#maxsizein = args.maxsizein
#bufin = Queue(maxsize=maxsizein)

pout   = str(args.vout)
bufout = Queue(maxsize=args.maxsizeout)
#maxsizeout = args.maxsizeout
#bufout = Queue(maxsize=maxsizeout)

#exp = args.exp
multiplier=args.multiplier

from interpolation import *



import cv2, _thread
from tqdm import tqdm

def reader(pvin):
    global vin,bufin
    frame_count = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
    #try:
    for _ in range(frame_count):
        rt, frame = vin.read()
        bufin.put(frame)
    #except:
    #    print("ERROR in the input buffer. Exiting...")
    #    exit(1)
    return
def writer(pout,codec,fps,apiPreference,params):
    #input()
    global vout,bufout
    frame = bufout.get()
    h,w,_ = frame.shape
    vout.open(pout,codec,fps,(w,h))
    while frame.all() != None:
        rt = vout.write(frame)
        frame = bufout.get()
    return

input("Press enter to continue") 




#vout.open(filename=pvout,fourcc=fourcc,fps=fps,frameSize=(w,h),apiPreference=0,params=[])

# INPUT
vin = cv2.VideoCapture(filename=pin)
fpsin = float(vin.get(cv2.CAP_PROP_FPS))

#help(vin.get)
frame_count = int(vin.get(cv2.CAP_PROP_FRAME_COUNT))
#w  = int(vin.get(cv2.CAP_PROP_FRAME_WIDTH))
#h  = int(vin.get(cv2.CAP_PROP_FRAME_HEIGHT))
#col = int(vin.get(cv2.CAP_PROP_CHANNEL))



# OUTPUT
vout = cv2.VideoWriter()
#fourcc = int(vin.get(cv2.CAP_PROP_FOURCC))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#fpsout = fpsin * 2**exp
fpsout = fpsin * multiplier
#print(pout,fourcc,fpsout,help)


tin  = _thread.start_new_thread(reader, tuple([pin]))
tin_complete=False
tout = _thread.start_new_thread(writer, tuple([pout,fourcc,fpsout,0,[]]))
tout_complete=False
#input()

lastframe = None
firstframe=True

for i in tqdm(range(frame_count)):
#for i in range(frame_count):
    #if vin.isOpened():
    frame = bufin.get()
    if firstframe:
        lastframe = frame
        firstframe=False
        continue
    bufout.put(lastframe)
    #X0 = torch.from_numpy(np.transpose(lastframe, (2,0,1)).astype("float32") / 255.0).type(torch.cuda.FloatTensor)
    #X1 = torch.from_numpy(np.transpose(frame, (2,0,1)).astype("float32") / 255.0).type(torch.cuda.FloatTensor)
    #X0 = torch.FloatTensor(numpy.ascontiguousarray(lastframe[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    #X1 = torch.FloatTensor(numpy.ascontiguousarray(frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    
    frames = interpolate_custom(lastframe,frame,multiplier)
    #print(len(frames),type(frames[0]))
    for frame1 in frames:  
        bufout.put(frame1)
    lastframe = frame
else:
    bufout.put(frame)


vin.release()
try:
    while bufout.qsize() > 0:
        time.sleep(0.25)
except:
    pass
vout.release()
cv2.destroyAllWindows()
