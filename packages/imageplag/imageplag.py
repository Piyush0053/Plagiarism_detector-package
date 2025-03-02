from plag import plag
def imageinfo(file):
    try:
        textpdf=plag.stream(file)
    except:
        pass
    
    dataimg=plag.extracttexti_image(plag.saved1)
    def readfile2(data):
        global listedk
        listedk = []
        sentences = data.split('.')
        for sentence in sentences:
            sentence=plag.process_list(sentence)
            listedk.append(sentence)
            listedk=list(filter(lambda a: a != '', listedk))
        return listedk
    def sentensiver(text):
        vf=[]
        for i in range(len(text)):
            bd=text[i].count(" ")
            if bd<14:
                pass
            elif bd>19:
                txt=text[i].split()
                txt2=txt
                while len(txt)>20:
                    ab=txt[0:21]
                    txt=txt2[21:]
                    txt2=txt
                    ab=' '.join(ab)
                    vf.append(ab)
                    vf.append(". ")
                if len(txt)>=14:
                    txt=' '.join(txt)
                    vf.append(txt)
                    vf.append(". ")
            else:
                vf.append(text[i])
                vf.append(". ")
        print("data processed successfully")
        return vf
    def splitfstop(text):
        text = plag.re.sub(' +', ' ', text)
        text =text.split(".")
        text=sentensiver(text)
        return text
    daimg2={}
    for key in dataimg:
        img1=dataimg[key]
        img1=splitfstop(img1)
        img1=" ".join(img1)
        daimg2[key]=readfile2(img1)
    daimg3={}
    for key in daimg2:
        img1=daimg2[key]
        daimg3[key]=plag.cerequest(img1, "text", 0, "lists")
    daimg4={}
    for key in daimg3:
        dc2=daimg3[key]
        daimg4[key]=plag.pass_datatext(dc2)
    rep={}
    for key in daimg4:
        dc3=daimg4[key]
        report=plag.plagreport(dc3, plag.pdfinfo)
        print(type(report))
        rep[key]=report
    tt=0.00
    count1=0
    rpp=[]
    rpp.append(rep)
    for key in rep:
        tp=rep[key]
        tt=round(float(tp['totalp']), 2)+tt
        count1=count1+1
    if count1!=0:
        plagist=((tt)/(count1*100))*100
    else:
        plagist=0.0
    print(plagist)
    rpp.append(plagist)
    return rpp
