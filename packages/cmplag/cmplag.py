from plag import plag
import numpy as np
def mutualchecker(file1, file2):
    def highbold(textpdf2, textpdf):
        detected=[]
        cm=list(set(textpdf).intersection(textpdf2))
        for i in range(len(cm)):
            ib=[]
            ia=[]
            nb = np.array(textpdf)
            na=np.array(textpdf2)
            searchval = cm[i]
            ib = np.where(nb == searchval)[0]
            ia=np.where(na == searchval)[0]
            ib=list(ib)
            ia=list(ia)
            for j in range(len(ib)):
                if textpdf[ib[j]]!=".":
                    textpdf[ib[j]]="<span class='bg-warning'>"+str( textpdf[ib[j]])+"</span>"
            for k in range(len(ia)):
                if textpdf2[ia[k]]!=".":
                    textpdf2[ia[k]]="<span class='bg-warning'>"+str( textpdf2[ia[k]])+"</span>"
        detected.append(textpdf2)
        detected.append(textpdf)
        return detected
    def worddoc(file):
        global error
        global docname
        global saved1
        saved1=[]
        global pdfinfo
        pdfinfo={}
        pdfinfo["CreationDate"]="hakunajibu"
        pdfinfo["Creator"]="hakunajibu"
        pdfinfo["Producer"]="hakunajibu"
        try:
            global text
            print(file)
            text = textract.process(file)
            print("doc text extracted succesfully")
            text=str(text)
            print(text)
            texte2=[]
            text=text.split()
            if len(text)>3000:
                for kll in range(3000):
                    texte2.append(text[kll])
                text=" ".join(texte2)
            else:
                text=" ".join(text)
            text=worddocclean(text)
            docname="/home/xz0r/upload/txt%s.txt" %(randname())
            text2=" ".join(text)
            with open(docname, "w")as f:
                f.write(text2)
                f.close()
            return text
            print("ok")
        except:
            error=1
            print("error occured3")
        finally:
            pass
    def pdfdoc(file):
        global error
        try:
            global pdfread
            pdfile=open(file, "rb")
            pdfread=plag.p2.PdfFileReader(pdfile)
            plag.pdfinfoo(pdfread)
            pdfile.close()
            global text
            text=extractpdf(file)
            error=0
            print("pdf sucessfully streamed")
            return text
        except:
            error=1
            print("error occured2")
            raise
        finally:
            pass
    def extractpdf(file):
        global error
        try:
            with plag.fitz.open(file) as doc:
                text=""
                for page in doc:
                    text=" ".join(text)
                    print(text)
                    if len(text.split())<=3000:
                        if len(text.split())+len(page.getText().split())>3000:
                            newr=3000-len(text.split())
                            appender=[]
                            ita=page.getText().split()
                            for kjk in range(newr):
                                appender.append(ita[kjk])
                            app=" ".join(appender)
                            text+=app
                        else:
                            text+= page.getText()
                        text=splitfstop(text)
                        text2=" ".join(text)
                        global docnamef
                        docnamef="/home/xz0r/upload/txt/%s.txt" %(plag.randname())
                        with open(docnamef, "w")as f:
                            f.write(text2)
                            f.close()
                    else:
                        plag.extractimage(file)
                        error=0
                        print("pdf text extracted succesfully")
                        return text
            print(text)
            return text
        except:
            print("errror")
    def splitfstop(text):
        text = plag.re.sub(' +', ' ', text)
        text =text.split(".")
        text=sentensiver(text)
        return text
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

    def worddocclean(text):
        text =text.split("\\n")
        substring="\\n"
        text=" ".join(text)
        while text.find(substring) != -1:
            text.lower().replace('\\n',' ').replace('\r','').replace('\xa0', ' ').strip()
        text=splitfstop(text)
        return text
    def stream(file):
        global error
        doc=file.endswith("doc")
        docx=file.endswith("docx")
        pdf=file.endswith("pdf")
        if doc or docx:
            text=plag.worddoc(file)
            print("word doc detected")
            return text
        elif pdf:
            text=pdfdoc(file)
            print("pdf doc detected")
            return text
        else:
            error=1
            return error
    textpdf2=stream(file1)
    textpdf=stream(file2)
    textpdf=" ".join(textpdf)
    textpdf2=" ".join(textpdf2)
    textpdf=textpdf.split()
    textpdf2=textpdf2.split()
    compared=highbold(textpdf2, textpdf)
    return compared
    
    


    
    
   
