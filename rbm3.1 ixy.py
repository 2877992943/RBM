import random
import os
import sys
import math
import numpy as np



inpath = "D://python2.7.6//MachineLearning//rbm//me-1"
outfile1 = "D://python2.7.6//MachineLearning//rbm//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//rbm//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//rbm//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//rbm//4.txt"
     
global wordDic; 
global docList; 
global classList;classList=["business","auto","sport","it","yule"]
global learnRate;learnRate=2
global alpha;alpha=0.2
global regularWeight;regularWeight=0.2
global numhid;numhid=50
######################

def loadData():
    global wordDic ;wordDic={}
    global docList ;docList=[]
    global classList
     

    ###################build docList wordDic 
    wid=1
    for filename in os.listdir(inpath):
        for c in classList:
            if filename.find(c)!=-1:
                eachDoc=[{},0,c]
        content=open(inpath+'/'+filename,'r').read().strip()
        words=content.replace('\n',' ').split(' ')
        for word in words:
            if len(word.strip())<=2:continue
            if word not in wordDic:
                wordDic[word]=1.0
                eachDoc[0][word]=1.0;
            elif word in wordDic:
                wordDic[word]+=1.0
                if word not in eachDoc[0].keys():
                    eachDoc[0][word]=1.0 
                else:eachDoc[0][word]+=1.0
        docList.append(eachDoc)
    ###total words in all doc ,including repeated words
    global ttword;ttword=0.0
    for wid,freq in wordDic.items():
        ttword+=freq
    print 'ttword',ttword
    ###################output    
    outPutfile=open(outfile1,'w')
    for (word,n) in wordDic.items():
        outPutfile.write(str(word));
        outPutfile.write(':')
        outPutfile.write(str(n))
        outPutfile.write('\n')
    outPutfile.close()
 
    outPutfile=open(outfile2,'w')
    for i in range(len(docList)):
        outPutfile.write(str(docList[i]))
        outPutfile.write('\n')
    outPutfile.close()
    ############
    print 'loaded %d doc,%d wid'%(len(docList),len(wordDic))


def ixy():
    global ttword
    global wordDic; #{wid:freq}
    global docList; 
    global classList;
    global wordList;  #1 x v wid list [wid,[wid1,wid2],[],[],,]
    widDic1={} #{wid:ixy...)
    widDic2={}
    ttdoc=float(len(docList))
    ##fea is 1 wid
    for wid in wordDic:
        interInfo=0.0
        px=wordDic[wid] #some more than 1
        for c in classList:
            #####
            py=0.0001;pxy=0.0001;#smooth  #px=0.0;already exist in wordDic
            for doc in docList:
                if doc[2]==c and wid in doc[0]:
                    pxy+=doc[0][wid]# pxy+=1.0 not right 
                if doc[2]==c:
                    py+=1.0
            ####
            pxy=pxy/ttword
            py=py/ttdoc
            px=px/ttword
            interInfo+=pxy*math.log(pxy/(px*py))
        ########
        widDic1[wid]=interInfo
    ##################
    outPutfile=open(outfile3,'w')
    for wid,freq in widDic1.items():
        outPutfile.write(str(wid))
        outPutfile.write(' ')
        outPutfile.write(str(freq))
        outPutfile.write('\n')
    outPutfile.close()
    ##############
    pair=sorted(widDic1.iteritems(),key=lambda s:s[1],reverse=True)
    wordList1=[i[0] for i in pair]
    stop=int(len(wordList1)/10)
    wordList1=wordList1[:stop]
    print 'ixy rank first half  wid ',len(wordList1)#len(set(wordList1)) no repeated 
    ###########combine pair[w,w]
    wordList2=[]
    for i in range(len(wordList1)):
        if i+1!=len(wordList1):#i!=j and no [i,j][j,i] situation
            for j in range(len(wordList1))[i+1:]:
                 
                ######
                for doc in docList: #if pair emerge in 2 docs, here record twice, should record once only
                    if wordList1[i] in doc[0] and wordList1[j] in doc[0] and [wordList1[i],wordList1[j]] not in wordList2:#[i,j]exist
                        wordList2.append([wordList1[i],wordList1[j]])
    print'w-w wordList2',len(wordList2)#wordList2 has repeated widpair
    
    '''wd={}
    for w in wordList2:
        if tuple(w) not in wd:wd[tuple(w)]=1
    print len(wd) #140 while wordList2 149
    '''
        
    ##fea is [wid,wid] #widDic1 widDic2 -->wordList
    for ff in wordList2:
        #print ff[0],ff[1],wordList2.index(ff)
        interInfo=0.0
        px=0.0001
        for c in classList:
            #####
            py=0.0001;pxy=0.0001;#smooth  #px=0.0;already exist in wordDic
            for doc in docList:
                if doc[2]==c and ff[0] in doc[0] and ff[1] in doc[0]:
                    x1=doc[0][ff[0]]
                    x2=doc[0][ff[1]]
                    pxy+=min(x1,x2)
                if doc[2]==c:
                    py+=1.0
                if ff[0] in doc[0] and ff[1] in doc[0]:
                    x11=doc[0][ff[0]]
                    x22=doc[0][ff[1]]
                    px+=min(x11,x22)
            ####
            pxy=pxy/ttword
            py=py/ttdoc
            px=px/ttword
            interInfo+=pxy*math.log(pxy/(px*py))
        ########
        #widDic2[interInfo]=ff# when the key same, the value mixed,-->only9 [wid,wid], not work
        widDic2[tuple(ff)]=interInfo;#print len(widDic2)
    #print '2',len(widDic2.keys()),len(wordList2)#lose featpair
    
    
    ##########
    pair=sorted(widDic2.iteritems(),key=lambda s:s[1],reverse=True);#print 'pair',len(pair)
    wordList2=[i[0] for i in pair];#print len(wordList2)
    stop=int(len(wordList2)/2);#print 'stop',stop
    wordList2=wordList2[:stop]
    print 'ixy rank first half   wid-wid ',len(wordList2)
    wordList=wordList2
    ####################
    outPutfile=open(outfile4,'w')
    for item in pair[:stop]:
        outPutfile.write(str(item[0][0]))
        outPutfile.write(' ')
        outPutfile.write(str(item[0][1]))
        outPutfile.write(' ')
        outPutfile.write(str(item[1]))
        outPutfile.write('\n')
    outPutfile.close()
    
        



        
    
def feat():
    global wordDic; #{wid:freq,,,}
    global docList; #[  [{},predict,truelabel],[],[]...   ]
    global classList;
    global wordList;  #1 x v wid list
     
     
    print 'fea reduced to %d'%len(wordList)
    
    ############ build v list
    nvList=[] #n x v
    for doc in docList:
        vList=[]
        for fea in wordList:
            if fea[0] in doc[0] and fea[1] in doc[0]:
                vList.append(1.0)
            else:vList.append(0.0)
        ############
        nvList.append(vList)
    
    ###########
    global dataMat;dataMat=np.mat(nvList)
    #################output
    outPutfile=open(outfile3,'w')
    for v in nvList:
        outPutfile.write(str(v));
        outPutfile.write('\n')
    outPutfile.close()
 
def initial():
    global empphv,modEpvh,empEpvh,modEphbias,empEphbias,empEpvbias,modEpvbias,modphv,modpvh,sampleh
    global vhInc,hbiasInc,vbiasInc
    global vishid,hbias,vbias
     
    global dataMat,numcase,numvis,numhid
    (numcase,numvis)=np.shape(dataMat)
    
    
    empphv=np.zeros((numcase,numhid))
    sampleh=np.zeros((numcase,numhid))
    
    modEpvh=np.zeros((numvis,numhid))
    empEpvh=np.zeros((numvis,numhid))

    modEphbias=np.zeros((1,numhid))
    empEphbias=np.zeros((1,numhid))

    empEpvbias=np.zeros((1,numvis))
    modEpvbias=np.zeros((1,numvis))

    modphv=np.zeros((numcase,numhid))
    modpvh=np.zeros((numcase,numvis))

    vhInc=np.zeros((numvis,numhid))
    hbiasInc=np.zeros((1,numhid))
    vbiasInc=np.zeros((1,numvis))

    ########all 0initial ,array not matric above
    #####now random inital
    vishid=np.zeros((numvis,numhid))
    for i in range(numvis):
        for j in range(numhid):
            vishid[i,j]=random.uniform(0,1)
    hbias=np.zeros((1,numhid))
    for i in range(numhid):
        hbias[0,i]=random.uniform(0,1)
    vbias=np.zeros((1,numvis))
    for i in range(numvis):
        vbias[0,i]=random.uniform(0,1)
    ########matric
    
    empphv=np.mat(empphv)
    sampleh=np.mat(sampleh)
    modEpvh=np.mat(modEpvh)
    empEpvh=np.mat(empEpvh)
    modEphbias=np.mat(modEphbias)
    empEphbias=np.mat(empEphbias)
    empEpvbias=np.mat(empEpvbias)
    modEpvbias=np.mat(modEpvbias)
    modphv=np.mat(modphv)
    modpvh=np.mat(modpvh)
    vhInc=np.mat(vhInc)
    hbiasInc=np.mat(hbiasInc)
    vbiasInc=np.mat(vbiasInc)
    vishid=np.mat(vishid)
    hbias=np.mat(hbias)
    vbias=np.mat(vbias)
    
def train():
    global empphv,modEpvh,empEpvh,modEphbias,empEphbias,empEpvbias,modEpvbias,modphv,modpvh,sampleh
    global vhInc,hbiasInc,vbiasInc #change paramenter
    global vishid,hbias,vbias #parameter
    global dataMat,numcase,numvis,numhid

    ########calc empirical Ep[v1h1] E[h bias] E[v bias]
    ####calc p[h|v]
    empphv=dataMat*vishid+np.tile(hbias,[numcase,1])
    empphv=np.exp(-empphv)+1
    empphv=1/empphv
    #print '1',np.shape(empphv)
    #####
    empEpvh=dataMat.T*empphv/float(numcase);#print '2',np.shape(empEpvh)
    empEphbias=np.mat(np.ones((1,numcase)))*empphv/float(numcase);#print '2',np.shape(empEphbias)
    empEpvbias=np.mat(np.ones((1,numcase)))*dataMat/float(numcase);#print '2',np.shape(empEpvbias)
    
    ######sample h
    for i in range(numcase):
        for j in range(numhid):
            rand=random.uniform(0,1)
            if empphv[i,j]>rand:sampleh[i,j]=1.0
            else:sampleh[i,j]=0.0
    ##########calc mode Ep[v1h1] E[h bias] E[v bias]
    #####use sample h to calc mode p[v|h]
    modpvh=sampleh*vishid.T+np.tile(vbias,[numcase,1])
    modpvh=np.exp(-modpvh)+1
    modpvh=1/modpvh
    ###########calc mode p[h|v]
    modphv=modpvh*vishid+np.tile(hbias,[numcase,1])
    modphv=np.exp(-modphv)+1
    modphv=1/modphv
    ###########
    modEpvh=modpvh.T*modphv/float(numcase);#print '3',np.shape(modEpvh)
    modEphbias=np.mat(np.ones((1,numcase)))*modphv/float(numcase);#print '3',np.shape(modEphbias)
    modEpvbias=np.mat(np.ones((1,numcase)))*modpvh/float(numcase);#print '3',np.shape(modEpvbias)
    #########calc para change
    ###grad normalization
    gradvh=modEpvh-empEpvh+regularWeight*vishid/float(numcase)
    gradhb=modEphbias-empEphbias+regularWeight*hbias/float(numcase)
    gradvb=modEpvbias-empEpvbias+regularWeight*vbias/float(numcase)
    gradvh=norm(gradvh)
    gradhb=norm(gradhb)
    gradvb=norm(gradvb)
    
    vhInc=alpha*vhInc-learnRate*(gradvh)
    hbiasInc=alpha*hbiasInc-learnRate*(gradhb)
    vbiasInc=alpha*vbiasInc-learnRate*(gradvb)
    ##########para update
    vishid=vishid+vhInc;#print 'vh',vishid
    hbias=hbias+hbiasInc;#print 'hbias',hbias
    vbias=vbias+vbiasInc;#print 'vbias',vbias

    
def norm(gradvh):
    [hang,lie]=np.shape(gradvh)
    ss=0.0
    for i in range(hang):
        for j in range(lie):
            ss+=gradvh[i,j]**2
    ss=math.sqrt(ss)
    gradvh=gradvh/ss
    return gradvh
            
    
    
def calcLL():
    global empphv,modEpvh,empEpvh,modEphbias,empEphbias,empEpvbias,modEpvbias,modphv,modpvh,sampleh
    global vhInc,hbiasInc,vbiasInc #change paramenter
    global vishid,hbias,vbias #parameter
    global dataMat,numcase,numvis,numhid
    LL=0.0
    for i in range(numcase):
        pvhProb=np.mat(np.zeros((numvis,numhid)))
        #####p[vh] mat
        for vi in range(numvis):
            for hi in range(numhid):
                pvihi=vishid[vi,hi]+vbias[0,vi]+hbias[0,hi]
                pvihi=math.exp(pvihi)
                pvhProb[vi,hi]=pvihi
        ##########z
        z=np.sum(pvhProb,axis=0);z=np.sum(z,axis=1);z=z[0,0];#print 'z',z
        #######p[vh] normalization
        pvhProb=pvhProb/z
        ##########p[v ]  v x 1 matric for each obs
        pvProb=np.sum(pvhProb,axis=1);#print 'pv mat',pvProb
        
        
        '''vi=np.tile(modpvh[i,:],[numhid,1]);vi=vi.T 
        hi=np.tile(modphv[i,:],[numvis,1]);
        wvh=np.array(vh)*np.array(vi)*np.array(hi);#v x h array
        
        bi=np.tile(vbias,[numhid,1]);bi=bi.T
        vibi=np.array(bi)*np.array(vi)##v x h array

        aj=np.tile(hbias,[numvis,1]);
        hjaj=np.array(aj)*np.array(hi)#v x h aray

        pvhProb=wvh+vibi+hjaj
        pvhProb=np.exp(pvhProb) 
        
        pvhProb=np.mat(pvhProb);print '4 mat',shape(pvhProb) ###matric sum works,while array sum not work well
        ##########above are model p[v],now compare true v'''
        pv=1.0
        for ii in range(numvis):
            if dataMat[i,ii]==0.0: #a[0][0] {} [[][]] structure, numpy structure a[0,0],a[0,:]
                pvProb[ii,0]=1.0-pvProb[ii,0] #pvProb[ii,:] is 1x1 mat not float
            ###########
            pv*=pvProb[ii,0]
        ###########
        #print 'p[v1=1,v2=0,,,]',pv
        ###### accumulate all obs
        LL=LL-math.log(pv+0.0001)

    ############all obs done,now regularization item
    hh=hbias*hbias.T;hh=hh[0,0]
    vv=vbias*vbias.T;vv=vv[0,0]
    ww=0.0
    for i in range(numvis):
        for j in range(numhid):
            ww+=vishid[i,j]**2
    
        
    ww=regularWeight*math.sqrt(hh+vv+ww)
    ##############
    LL+=ww
    print '- loglikely',LL
    return LL
                
def cluster(modphv):
     
    global dataMat,numcase,numvis,numhid
    global docList

    cluster={}
    for h in range(numhid):
        cluster[h]=[]
     
    for i in range(numcase):
        trueLab=docList[i][2];
        maxL=None;maxP=10
        for j in range(numhid):# 0,1,2,3,4
            p=modphv[i,j]
            if maxP==10 or maxP<p:
                maxP=p
                maxL=j
        ###########
        cluster[maxL].append(trueLab)
    #print cluster
    
def feaMirror(empphv):#p[h|v] n x h
    global dataMat,numcase,numvis,numhid
    global docList
    #print empphv
    #applicatino:use new feat calc distance
    for n in range(numcase):
        true1=docList[n][2];print true1
        simi={}
        for nn in range(numcase):
            if n!=nn:
                true2=docList[nn][2]
                dis=0.0
                #######
                d=empphv[n,:]-empphv[nn,:];#print 'd', d
                dis=d*d.T
                dis=dis[0,0];#print dis
                ##
                #simi[true2]=dis #key:true2 is repeated not unique
                simi[nn]=[true2,dis]
        #########
        simi1=sorted(simi.iteritems(),key=lambda a:a[1][1],reverse=False)
        print simi1[:3]
            
            
            
    
    

    
    

###################main
loadData()
ixy()
 
feat()

initial()
###
train()
LL1=calcLL()
LL0=10000
i=0
while i<=50 and LL0-LL1>0.02:
    LL0=LL1
    train()
    LL1=calcLL()

'''
### apply the probabilistic into practice
global empphv,modphv
cluster(modphv)
cluster(empphv)
'''
global empphv,modphv
feaMirror(empphv)
 
    
 
        
    
    
    







    
 # loglikely down to 720.218571356  learning rate=1
 # learning rate 10, loglikel not always down , fluctuate
