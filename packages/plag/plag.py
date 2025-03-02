from IPython.display import display, Markdown, HTML
import PyPDF2 as p2
import fitz
from googleapiclient.discovery import build
import random
import string
import pytesseract as tess
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import rabin_karp
from rabin_karp import rabin_karp
import numpy as np
from os.path import dirname, join
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import requests
import textract
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
error=0;
doc=0;
def pdfinfoo(pdfread):
    global pdfinfo
    global error
    if error!=1:
        pdfinfo={}
        try:
            enc=pdfread.getIsEncrypted()
            pdfi=pdfread.getDocumentInfo()
            if "/CreationDate" in pdfi:
                pdfinfo["CreationDate"]=pdfi["/CreationDate"]
            else:
                pdfinfo["CreationDate"]="hakunajibu"
            if "/Creator" in pdfi:
                pdfinfo["Creator"]=pdfi["/Creator"]
            else:
                pdfinfo["Creator"]="hakunajibu"

            if "/Producer" in pdfi:
                pdfinfo["Producer"]=pdfi["/Producer"]
            else:
                pdfinfo["Producer"]="hakunajibu"
        except:
            pdfinfo["CreationDate"]="hakunajibu"
            pdfinfo["CreationDate"]="hakunajibu"
            pdfinfo["Producer"]="hakunajibu"
            pdfinfo["Creator"]="hakunajibu"


    else:
        print("error occured1")
    


def randname():
    tmp_name=''.join(random.choice(string.ascii_letters) for x in range(10))
    return tmp_name

def worddocclean(text):
    text =text.split("\\n")
    substring="\\n"
    text=" ".join(text)
    while text.find(substring) != -1:
        text.lower().replace('\\n',' ').replace('\r','').replace('\xa0', ' ').strip()
    text=splitfstop(text)
    return text



def sentensiver(text):
    vf=[]
    for i in range(len(text)):
        bd=text[i].count(" ")
        if bd<14:
            pass
        elif bd>19:
            txt=text[i].split()
            while len(txt)>20:
                ab=txt[0:21]
                txt=text[21:]
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
    text = re.sub(' +', ' ', text)
    text =text.split(".")
    text=sentensiver(text)

    return text

def sentensiver1(text):
    vf=[]
    for i in range(len(text)):
        bd=text[i].count(" ")
        if bd<2:
            pass
        elif bd>19:
            txt=text[i].split()
            while len(txt)>20:
                ab=txt[0:21]
                txt=text[21:]
                ab=' '.join(ab)
                vf.append(ab)
                vf.append(". ")
            if len(txt)>=3:
                txt=' '.join(txt)
                vf.append(txt)
                vf.append(". ")
        else:
            vf.append(text[i])
            vf.append(". ")
    print("data processed successfully")
    return vf


def splitfstop1(text):
    words = set(nltk.corpus.words.words())
    text=" ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())
    text = re.sub(' +', ' ', text)
    text =text.split(".")
    text=sentensiver1(text)

    return text

def rm_dup(accepted, indexpng, imgstr):
    intext=imgstr[10:]
    nextens=len(str(indexpng))+1
    unique=intext[nextens:]
    dupst = any(dup in unique for dup in accepted)
    return dupst

def extractimage(file):
    j=0
    doc = fitz.open(file)
    d=0
    global accepted
    accepted=[]
    global saved1
    saved1=[]
    for i in range(len(doc)):
        for img in doc.getPageImageList(i):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            d=d+1
            if pix.n < 5:       # this is GRAY or RGB
                tmp_name=''.join(random.choice(string.ascii_letters) for x in range(10))
                imgstr="%s%s_%s.png" % (tmp_name, d, xref)
                if rm_dup(accepted, d, imgstr):
                    pass
                else:
                    accepted.append("%s.png" %(xref))
                    saved1.append(imgstr)
                    pix.writePNG("/home/xz0r/upload/img/%s" % (imgstr))
                pix=None

            else:               # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                tmp_name=''.join(random.choice(string.ascii_letters) for x in range(10))
                imgstr="%s%s_%s.png" % (tmp_name, d, xref)
                if rm_dup(accepted, d, imgstr):
                    pass
                else:
                    accepted.append("%s.png" %(xref))
                    saved1.append(imgstr)
                    pix1.writePNG("/home/xz0r/upload/img/%s" % (imgstr))
                pix=None

            j=i
    print("number of page is {0} and number of image {1} saved".format(j+1, len(saved1)))


def extracttexti_image(saved1):
    global rmi
    rmi={}
    for i in range(len(saved1)):
        img=Image.open("/home/xz0r/upload/img/%s" % (saved1[i]))
        text=tess.image_to_string(img)
        text=splitfstop1(text)
        text=" ".join(text)
        text=text.split(".")
        text=" ".join(text)
        text=process_list(text)
        rmi["/home/xz0r/upload/img/%s" % (saved1[i])]=text
    rmi={k: v for k, v in rmi.items() if v != ''}
    return rmi


def extractpdf(file):
    global error
    try:
        with fitz.open(file) as doc:
            text=""
            for page in doc:
                text=" ".join(text)
                print(text)
                if len(text.split())<=533:
                    if len(text.split())+len(page.getText().split())>533:
                        newr=533-len(text.split())
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
                    global docname
                    docname="/home/xz0r/upload/txt/%s.txt" %(randname())
                    with open(docname, "w")as f:
                        f.write(text2)
                        f.close()
                else:
                    extractimage(file)
                    error=0
                    print("pdf text extracted succesfully")
                    return text
                    
        extractimage(file)
        error=0
        print("pdf text extracted succesfully")
        return text

    except OSError as err:
        error=1
        print("error occured", err)
    finally:
        pass


def pdfdoc(file):
    global error
    try:
        global pdfread
        pdfile=open(file, "rb")
        pdfread=p2.PdfFileReader(pdfile)
        pdfinfoo(pdfread)
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
        if len(text)>533:
            for kll in range(533):
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

def stream(file):
    global error
    doc=file.endswith("doc")
    docx=file.endswith("docx")
    pdf=file.endswith("pdf")
    if doc or docx:
        text=worddoc(file)
        print("word doc detected")
        return text
    elif pdf:
        text=pdfdoc(file)
        print("pdf doc detected")
        return text
    else:
        error=1
        return error

def readfile(filename):
    global listed
    listed = []

    try:
        fd = open(filename,"r")
        data = fd.read()
        sentences = data.split('.')
        for sentence in sentences:
            sentence=process_list(sentence)
            listed.append(sentence)
        fd.close()
        listed=list(filter(lambda a: a != '', listed))
    except:
        error=1
    finally:
        pass
    return listed

def empty_listrm():
    unique=' '
    dupst = any(dup in unique for dup in docdata)

def process_list(text):
    substring="\n"
    text = text.lower().replace('\n',' ').replace('\r','').replace('\xa0', ' ').strip()
    while text.find(substring) != -1:
        text.lower().replace('\n',' ').replace('\r','').replace('\xa0', ' ').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]','',text)
    return text

def apikeysconf():
    global api_key
    api_key=['AIzaSyD9pcLKtYLn3YuSE4qVS12lB5viCj6NoiU','AIzaSyCPVN-lpAi5cFl_Mv0W300-o4f2R--QLic','AIzaSyAIxcdz6U7NgJDpm5fCxBhknqD3Xg0llUY','AIzaSyDj0Crz298grB5WcYyOmz2C65BtgbUWtDk','AIzaSyDXn8PQM8x0jbpcEs-DnQ76jWCDS7END3U','AIzaSyDHH2bBx8CZSVsKvl4tyVub-Z624vYCTYs','AIzaSyD_DADq3qthcDILRqCVgJmaDC1HhhK1R7A','AIzaSyAii1Edcl9VcCE3YJNResk6ZjCyfVbYH1s','AIzaSyD-bdZ7DRnrqJF1u_VCUzS3UnsGeKyGi8w','AIzaSyApRjKtg8aun_S9rqhy4Dg8EP_URCVUgDw','AIzaSyCaK7sN6pHbm1wBlFakYLOV7iOzem-ovmo','AIzaSyCcTrDCvmG_09tGQo38v5WYpa6YD83IsK8','AIzaSyD9NVJJvwq4gtOB_r5KhQAjTEd2l9MFMmQ','AIzaSyB3TqvB7gVsI6jAwHzrocXqIO2A7mskgP8','AIzaSyDRtbuDKXir1ykveV-W4kV5HJ8uxDvmoBo','AIzaSyC3Kq48ings8n2IvVhZGYbhYZpiQ_ec7z0','AIzaSyAEnNthcI0EJjZ99356iUufKg0qfa5XOlU','AIzaSyCmqtMI9kntqquqfdSy7cCgLIHp7Vm5cX0','AIzaSyCuGnbAa0DoWChetn2Q2OIODZDYW2ShAnc','AIzaSyCdyiIwn_RzGUKdF1WLf0kCEGoUFAd_mPg','AIzaSyDqd0ZvMKi4a_oOHNXjRu9bVjsljmpcskE','AIzaSyC0sGGREiIPSD00E4nPqBLqL4mrb6zOu9w','AIzaSyC_P49nr7-_USnyiKaYJQDT8z7qJunqXjU','AIzaSyDoJRDmWU_-eLlGCHYtzGy4DhtzdPtU6nc','AIzaSyD6tm5kMCecP_RqxHmZCYlUxj0cy8mZCpo','AIzaSyD0U7BahUR_dY_2hV4mZE3fQrWETqy_BnA','AIzaSyBq9VlriHumD0V3-qTA7Amv87zyI3nOTO8','AIzaSyDRjcKhtlZblT6XjeEWgPDxrXYsZfTxrwI','AIzaSyDJZ29naWKimEaY44KDtryTCuF84GnZ1Xs','AIzaSyAHlDQ_Lz__VLmF3IiNCg9XvzmZHlAaEBw','AIzaSyBGDclmVdhQyOLpPoQ-xnai3rPubN1gKsE','AIzaSyBf_c3vb9GVehldQANfoy_Fn3QU-UHXGsA','AIzaSyDNO2yoxVyn02fXrYGcQPUHHF-3eo5Pw9M','AIzaSyAZ8npHFM83EnZH6-H3201Y4zOIOijBwO0','AIzaSyBqoU08cR6OC2SfQINfojLA-gDOrkUDBNo','AIzaSyBmpClVvvCy-f0jaP-8X3bQbkM2jszD6hw','AIzaSyBhRY6brnkJrEmVsptbD5DY1Wr900cKpC0','AIzaSyAkkqVR7jHp9BrvkmkHlx408ZOuxGzORBM','AIzaSyBoHjPAmKfXvyH6a6rIGZ6Q7SQOWISnjvI','AIzaSyDKAfZ4YdubZvQXJ4hy7-3c6sqUIGv9At8','AIzaSyB9U87FqF3Fon6uBxh7RzeJba3AHQmOF8s','AIzaSyDf8dqGMxkS2B5SToBZne7i9JpOkzFiIDI','AIzaSyCjOapy2RJfewqSHQFYDPfLpkH-4FRy6lo','AIzaSyCTaT5N7H6oTefGYwsuI2V4TYkAHeEw8II','AIzaSyBKV9QMAduVBaEjtSvdIqAI8iGBbqNxI9A','AIzaSyDyhn5bUn7NMfISYOgwwYK-9IS8hasEwPA','AIzaSyCE5GAzCEp7fEPwqm7k4vG_X_ut4iVIRCs','AIzaSyDXSVjdOGRlycE7mtJ6ZH4zd7LHITCBzxg','AIzaSyCi7e-P_RTqBoyEP18rEQQuIKSC9sTRpSA','AIzaSyD2EyxDM6Vu_daJU0Y1BR9GalclIzguW-I','AIzaSyBjJcqvYbHJx3WMzLapPIJEDkTW_Bm4XBg','AIzaSyCr-IDi9cX9x4OL1t4tcL5yh-rICxVtuXw','AIzaSyCceSAo14nfMuMVcDi94-9gZVDGI3rGJk4','AIzaSyDnvPtpFwWSIfJirOmYEM_alcyBtbbEdYY','AIzaSyD4C6Ygk9hH2MJSucSFdLUvy0KLQzE-4M8','AIzaSyAYdqYc95X7CbPjB_I2xHq_VyB_V0dCwWg','AIzaSyBY-dXoSB7reIJ5l6ltRICwxEY_gNC-Eoo','AIzaSyAFN8B5EKMiqCudmDBGFdLOLegQI60sb0o','AIzaSyDWvis58yjnERthXIp9QqV_AcIaIV3sbYI','AIzaSyC4Ny2AaRKGgJwGz6t9ulUi5ECUnrzmev0','AIzaSyCHSHTqtaIqTaQCOceYnEy5fpp0HmvSLCY','AIzaSyAZzG4GqtTgKWgA41_-ZvcQp0YlaBXddkc','AIzaSyDIBRDPu-wkMuVSxizbsC9nU2_N2vNJHHI','AIzaSyDcP9e6toiq_RTKUaPwDQWDx2opYHc0ZrA','AIzaSyCiWRkT2zr35sQFMk-dktS_A14K1ypoixY','AIzaSyDp7mUUl2E5EWRO2kh5GlPkVAHHOxkmh0s','AIzaSyBNvZUYcu1Sfuq649L63Z1rpC0QX4qaLLo','AIzaSyAwsiLuyaHCdQKNZYtBYzzyWoeXnUpZ-IE','AIzaSyAho-RU5y3I1iy_8CjECJQVIebKwEL_U3o','AIzaSyDxEfcKrRxUl0TLc68FOOt3Hm3X_JgXopU','AIzaSyD9TWHTbZv0BgRJ5GCYSoKmHvLbvA7Tt94','AIzaSyD9hdkjtQvamzXknpElsU_dguAZwVA2INw','AIzaSyAHgqSXNthl1nl1IweSiRSjjxqE965fQNs','AIzaSyDhKBWrx3iX_Rnis6GVw0GFZ02BOs1xBR0','AIzaSyDYgqAA-_-e-VGqBa5ORIqrUNO7GElYhAM','AIzaSyA-eZs_yJDMNoLkJexSTJbA-9ro3ZACtQI','AIzaSyAm7yn607Y-I4bEbLKL9hIw2U93JXyuWKs','AIzaSyBffxYxKZyqIihJ1gxQUbzk3qzRoUU-J6s','AIzaSyBCJWo79wsy7_k8nl__2g-W-ITNMPfK3eQ','AIzaSyDxfgaGPziWUMAr5JENL4OtnllrqfdwNoY','AIzaSyDXXwpcmV0J0GUyZUiGZN3gSANjtbD2jXY','AIzaSyD2Uw1z8BV9XXKMG--as3LDbL7aZndfat8','AIzaSyCFtvLUIc0oJQ0DuFpFk6iQuCE4R70kzhw','AIzaSyC_PPvi3So5fZx2hqkUL10a95-sfgw4sIM','AIzaSyAHbfNqizhYNXfBvIRPBCeABtZvaxsq8Ks','AIzaSyCWsDBh9lxVIPSxPuHxNRLwjTJ8D4bgIH8','AIzaSyChDWZ_OvMaseUxM_y7yV18FfJFWJF-uUg','AIzaSyBRXa1Ng9Z5YZUrPZp0FM3SaMc1K7a4dLE','AIzaSyDcUDgCS8txfrbiFLH4-CC2b0h3kHsCDQ8','AIzaSyDg-qA-4AyMbORrsmTBn2wNS7olZQK0UlY','AIzaSyCi5UdVoqDAqatVecllYF5JW8BYfeyV4OQ','AIzaSyA7wkmA__jIt0_wdHIcGxx4B74XNZMDOK4','AIzaSyCECMoRRfnhkAou2Fs29JKbT4-9vm0uyVk','AIzaSyB6i03boo6UkjyiYKH6pVZkViKwKHJNte0','AIzaSyAIXPOY4aXFi8VAsk9NqL4dEiNC6AbnRaw','AIzaSyAO3Y9WfyGlA0GHpOBXgz5uwqCdYhJhtVM','AIzaSyDKF9kSAh7R3ASypSd5LZYtu5yma5Dlyic','AIzaSyBZmW6ZJkGVHlxyA0PxPP9LKxHodZXDZ0E','AIzaSyCmIwmLl-JOr2cSR9lLF1QhzkHvK2BCbgw','AIzaSyBNAr31YGAcSO4e61TXPG24OJ5XoV885LU','AIzaSyAWCDyYZUKBAAJgI4EsI1YYlAj7_EDrs6o','AIzaSyAQMFVv8564fqmB7B4vpwTfFg-okqXe4CQ','AIzaSyBL9zDN8Yd6DEEcY-yZOBK_oUomubW9vjo','AIzaSyBCI5KIs3CmiAEg1GlJZmpphUVPE8B4g-g','AIzaSyBhE2-u9thSOd5w2l78irkbOdJ86EIFHq8','AIzaSyA_R6V11qNJST6l_lb4nE9UsgNlNQ_A5H4','AIzaSyAu5Z4w8n7zGDELFlHnBsuRhqXatjAYM2o','AIzaSyDGzfn9FcVCFak5HHGrENxYWRQYYdpjph4','AIzaSyCKtVJ6SEBEA2_OFy6kMUjGauJspRln2ts','AIzaSyCXoEKh8xdho5tnmO1ktentWatyCc2SyZY','AIzaSyA6C_IajmA7zcY3sV-s_UZYZ-z6opkfFO4','AIzaSyDIz2Z1tS41GiT10BwoQ8x8iGwzZqnS5UM','AIzaSyAyFFpZohgffa_pb3X04W27l_yAAP3Rn9M','AIzaSyDxboeI4veRtIoMEKG8UtWyH22jugq4zZc','AIzaSyBVLjqW2GqNA9aM87r564Yk1Rx7HuV3Dho','AIzaSyBHur7qMHZ5Z68aiievr3HihijeVqpFtEk','AIzaSyDnCZJpNJeYRskiZoF5oGJ3xPJfW_Iu3Io','AIzaSyBsu1OhiXsd07YLZ9qynB0aRD2Q-RHRDns','AIzaSyBSkBbKS3-MVTegqJPI3UG04o-YCErrEe8','AIzaSyBvOOcytsHNqm9lPXaPAbdUXA3PswW71hs','AIzaSyDL2UYlwE_Cp2_AWca3zjQDNsa86SftN2c','AIzaSyAsR-_xyuj8-HYdOCAhYmQMsomGZoZswUg','AIzaSyBtyHmTw_zxyGzafpnrNpWFa-7Fgw-KYZo','AIzaSyDUfwyrP7bcG4IPZRqKlIcWsTpF3NxKhyE','AIzaSyAq5Bbbqr_iYbeWvcO2HgGO5NoDc3VDZB0','AIzaSyA5wPkj2U-FIrk0u_ZU2HfL8m-tz2UPZgw','AIzaSyA6pqlyaRkkWpzz8iC7U9czopAnJ_kdmgc','AIzaSyDwUuUISNk7wjf-YDNcPYg0GxFiFeMHimI','AIzaSyAMabAk_19pu1J03cDvvNnpUIs16bNXx-Q','AIzaSyDx1_wsNbN3AS3rWhrAiTxmHX268eWCF8Y','AIzaSyCvh4ay4-gS401DhwEVvPxK1F-aXYtwwIk','AIzaSyC7jFQtVG2QPwkqA5rwwAuK6gdj2bldsrI','AIzaSyCDzXzY1miIjFRqWKRh-kOxbEOGCtGHumQ','AIzaSyCQF30k_Gt-HZCdBVqB9mlurQ4tg8aW5Rc','AIzaSyBDDg6WLSNXok_AJsyH5B10k1v2ktBPqIM','AIzaSyCuFrOsf1uFyrSLe4R8l8ON_8dQQMfoDPI','AIzaSyC0npkwvbZB8UE3A6F-htJXqsokNJ8pfaA','AIzaSyAwhwr3_04gcR2aZB3TE82gdbQ1OFmNWQc','AIzaSyAFT1i7nO4eMRZ2PruwTq5h4QLeTf45p8c','AIzaSyBP-GbZr6T3eK_-Dv-hyR8eJbWAO_-c5QI','AIzaSyBKot-tgEtscM0n088GJtFUhGrrvE5UEKc','AIzaSyCoQjw9mdP-2iG_r0GYezmnS0Qcp7xXKto','AIzaSyB3KHDYqvzeSuuXucJgN4PIalJGZqQo3v0','AIzaSyCRdlwu-2g2UjAQqNojYBkw4TZ8w2fn83o','AIzaSyBehWSd9co3ZWZDroQu0he1-utK8gP9iRg','AIzaSyB7vT_x8n4-9jUGmY7FWEb5GBeuwzIuttI','AIzaSyDiqnzS5T1FKrc3_UsKBxifT6-PmMmOIQU','AIzaSyC2Q2gVD1nkF-zZ0VqZP6yc6rC-SSEGqP8','AIzaSyC3phU94gdYcanz4igyKuvefOZ_fQM-JWg','AIzaSyBVpktWfGM2FwI9rhko8SG7OVT3MToo91A','AIzaSyDdNe8fIT6n_heoAK8FRuQAAvRhUfEw3rg','AIzaSyDQwIf61vsdcy01zRVTKK6uxC_qaYzAb10','AIzaSyAIYacQOc5vrLtduH3icyRm07C-5P-AuYI','AIzaSyBXsyQWBbaJ66t4JzaWHt4xh6XMTn3RhLY','AIzaSyDYEEZQaYupmvDxha9h6W7j8XkStw552V4','AIzaSyDsEAUmYl2V4hCyYe__FI1bgHT6Lg4M_gw','AIzaSyDxZGBbrBDrOov4IlacsV7Q3dm2_oftYTc','AIzaSyADps2P6YOuhLv7lc-izBwK8dkO7deO5wI','AIzaSyBnRjF3vw8PvLQKr9W35D6kesqobPTkBSM','AIzaSyC4DGjqlRuNaldWW5YK1NqEEfhpNHBrHF8','AIzaSyDTakKBkPfXIDkZLqEnJ6ugWud9MRzlZXM','AIzaSyDzL0MBsmVEGfUq67-5gTzINBSJYcKWI44','AIzaSyC66VZtLyVp10zA5jsaSataIBK-e26Fp6A','AIzaSyBLFUYyyvH7PiKSXQYO16uPvBxY4G7XTZA','AIzaSyCUHdM1d9fYMej90LN1u2ETT2jTHjn-TJc','AIzaSyCV-MkJSynZUWnKfjtdHU2hoyN42wQqiHs','AIzaSyAaYQ00NTK9ygOsB2AifXDOCpurNL0YOLQ','AIzaSyC7zXqwgdJjp3zqMGhd0Mhn4RUeNp-Zaxc','AIzaSyBnj-4qlsZLEmv8YTtDPzjazsXygn5H32Q','AIzaSyD66QCBqtZfQNL2mNef3yX0X0bxfQdmkMQ','AIzaSyDGSq6Kqao8R4LbK19bejAExGolA6K0I3M','AIzaSyA-60GuAKWz02jw98D64wXrrf0D-dknsw4','AIzaSyAewR81dY79dkLba0KkXuzHY3DvIQkXhg8','AIzaSyAY2hJI01HSK_u9cGkEVlkmpQwEpYMjNIg','AIzaSyDqAMs4NiIdJa3CYvjfiwKf6zSYuLKAvmg','AIzaSyC12ESozVOTYNQ0bE0Szs0B1uaZYtrf-WM','AIzaSyAFTD5AaDVazQY35sQsx1WneOp19LJvMEE','AIzaSyBRsTXVCGUoPiuRkOfRJTWOAawZBtsKDbQ','AIzaSyDad2e9bKNWK0Yj9zTILUgXRb1vpHhSmRM','AIzaSyBauFo69UiPUYilphvRWI5u0ufevTBrJiQ','AIzaSyCDqhkS-ivmjJ1hvSYU6hmHa43rNe1qL9o','AIzaSyDwmDsneULTJprSwQF-qaD2JATQsUxIPLQ','AIzaSyAiBcSA1y9f8WY2H4ZmcvihiyQ7gjsYen4','AIzaSyCOtskLEu7eDsSeOvORzXoF0wcegWDEMHw','AIzaSyCZp8Qrw2ExR5YRNPW6nBkPgJRoEse3hJU','AIzaSyAan_B4NGbMBRfIR1AJN1PYNt1X4Xjr_yw','AIzaSyDUMkfmKzkSdYhw2XHHY6o-vHL6Qq2ifR8','AIzaSyCfAx6y9bf-lA3yeDbUeJGwF5mPu_8Td-Q','AIzaSyAPnWkcO4zd82WPzkAYe8MDzjXnWcv20LI','AIzaSyAn-MBXd-9UaG94BBoKkdoZfps0HT84cTo','AIzaSyBrttgWU89M3QukSLCFVT2BshsB5eucZM8','AIzaSyDflNA9ba2lL6O4_s3hD5IV96eXFaj8q5g','AIzaSyBIl5axdl_l4gAgjwwI7vGdMmixV1piowg','AIzaSyDdDEVK5ok2vUt1RE5g9pmEt_u0wPjK0RM','AIzaSyCbB6m6T8CxLE7UQOzrPu-L9cUj1ZhY_IA','AIzaSyAf2j6DkeP9gVErY1hRXGZhorg9vhnAPCY','AIzaSyAzkQf7SZZRLUbSRcXGkQfz_T3H_orkQ0Q','AIzaSyBklr6KKDO0Swxu3UMv1S5b8WoeGQs_dWs','AIzaSyBTdiHaWZ9DRXC7UouAsNppCZptmuqySXI','AIzaSyAzhELHDMQFWjn9jkHxapPqsGO882ZjBuU','AIzaSyCZpAHzinQOUMJ49vUjl8cZoTkzmdbFhmg','AIzaSyCZpAHzinQOUMJ49vUjl8cZoTkzmdbFhmg','AIzaSyBPsS8lLup38iTOGeKGC6UhOUP4oErH5AQ','AIzaSyDvKla4olDuGFZYCzRydPqSf_29cXtfO2w','AIzaSyA41_9zU_jD8UpvVrJR4noZa2yMcgEnYqc','AIzaSyDu4gDncKTaozWKgISFsmOr0OpfjPUlTcE','AIzaSyB37f0GvpFK5VwkhQhGQkzO6RzXEwcgCLA','AIzaSyCr_dxmcm-YZg8pc7EKGFsSHKJOr4_eMRM','AIzaSyDgMoa0xi-wEGPcImRJwd50DHf5dG27drw','AIzaSyCW_MS7NReUe-dTR0iFPfcw0uH5S8hkjp4','AIzaSyDjFUCqekXDfTSNIDmkVdvYJuA4GOmH7zY','AIzaSyCOUHOz9gQr_do0BdwPMwRbtVuxMlMzqMQ','AIzaSyB8CJVOXbecCb7W-6z1iF9xSwscP2j2D9c','AIzaSyCbm-6aod4sGIm_pHry0VcQP7FkCnhw3Ac','AIzaSyDFHDyVy4Q3lB6tu20aSZkesJmr9Pjio_g','AIzaSyBFeQu0uypd1ZqZdnir4TN-uWCGrldJUyw','AIzaSyD5R3s0qe6wLhfBzG7Gb1PFWCFidxvvjeA','AIzaSyCKAXr14rjsJ2x9aBdG6SOsK29COHexYso','AIzaSyCzYgUGFrCfYL1I4dozpo79pd4OWH93Y98','AIzaSyC-G9eIOKYh53CX1ztNkGaM6sbhiLRdqPc','AIzaSyBqIKE2srEulRPH1nYtJs90Rn9l28jC-Xc','AIzaSyA33CFx4sfs0kWnpwPZSiuw_hE4D8OK0C0','AIzaSyAZ1AoVtbn8P0GYR2LJOI5I0QKa589TPm8','AIzaSyALmpM_2_LMUq6dDp2vemUOEhfwNQrob_s','AIzaSyALmpM_2_LMUq6dDp2vemUOEhfwNQrob_s','AIzaSyCFHKsr3PFiwp6dXFUWS5yhDhCLnk5-pFk','AIzaSyD1h8ln0_ESW52gteLyLXNa71OQkXpxv4c','AIzaSyD1h8ln0_ESW52gteLyLXNa71OQkXpxv4c','AIzaSyDYfAcdoKNGWNZhXZihWTRj8MSiyKgzqrY','AIzaSyDH0X_zOjjWqHYnEh2Cjezy0M4mjvECX_k','AIzaSyBSVFKZB4lBlauU26flsdibIAjH6WZgF68','AIzaSyBAxprtA9xrEzsJDKglglsPP6q2wHoQJXM','AIzaSyDiKE_VYJd6c10dzD-ckFhFE-OQn7pgs6k','AIzaSyBzVZZmwsSx9vxqbwktgI-JI6qSIJKCpXM','AIzaSyDr9Wn-kUafvgTTBoD7khQoDq0oUKLe9vM','AIzaSyC_CM-6-P8wczYg8J9tmZ9uzSCoecJpZ9k','AIzaSyDILCD5Jyx8H0bvr-ndKDJ5a7kjgx6olsk','AIzaSyADHPqcRY3G5ti-hS52EHp3Y3O6grKFNjQ','AIzaSyAV_kMSMJW4Cq8FcES6mi25foIJRnWs30Y','AIzaSyBtzlrfnV3x4NCpxbEKtO_7is9G35QhNOM','AIzaSyDJDCVPmltWCTMZc9xEQpJrizk2BEWbm5k','AIzaSyC-kw4wONXyZetMR5qxhxI_5qziL5i-55E','AIzaSyDGCQ42w2ViMgbsboO54B9LuxLTLUIsIuo','AIzaSyAcscUqNYc5oYGtkpCP_Jh5W2lhL7ElSuE','AIzaSyDK85ySbHXf-0KeP-mzjh_RqEyVR9KfftY','AIzaSyBnXztLlhCxj4TPUOhq6DVKiv1Bg8EqlA0','AIzaSyDCN1BaPa2EXi6kdbHyK_7b6YdA9m1iAH8','AIzaSyA9WtIshSRjN40PVuX4V0Ymg8pNPN-Z_CQ','AIzaSyDBGkJujgV6es8DFkga370yje-Wa2_TFWc','AIzaSyBZmfpQ5c7l894cU5Kf_D4wb-EzXJIq_tk','AIzaSyA6EhyU01SI7CbPIk92i_R04AQctFyME3k','AIzaSyCU4Co1hX7V-g2mTgCvnY-NmG17CncGrkM','AIzaSyAjf8-ZPr5oXwRn2zLIwa2_hXBP9akBWZ4','AIzaSyDjwX7txSX2kf6EQ4NH_wOPgwmbUpM_f0I','AIzaSyCO2AnGq2TxcDyyCHVPf8oTtlpa29mIkek','AIzaSyDTe18hdAcgi8QmeMrFkrxPoZlVKhw6OOA','AIzaSyD1gqHATWJEPQo-itl_CfL4rqbVFSQkqec','AIzaSyDPDwyNu3-KKCzS14zyWgcxEbZhjSPW4VE','AIzaSyBSjNmFCQbK6TL_IhJmqve3DvxINLQiPZs','AIzaSyA4v-NpJ9Exlkco_s1K7OLTArdwGsUnarg','AIzaSyB7nu2QHBXlv-untOCLZOVtcjUPFq7K6RI','AIzaSyCM6K3MZYzSpIaanFXqGmmeO6TbOH5Lw9E','AIzaSyBfKvpmEGcTOGAjSXgLKRI-HU4uCVJRi7E','AIzaSyBr1amp3Cb0B6RfKb7SkCQEkcx5QI9gX20','AIzaSyCZIiowinwH5kzvPr_YTn7gTMKepBMKx9g','AIzaSyCo0CAIdBX0phJFdX1FO5FFqTdiV-FpFgo','AIzaSyD-k8Hg80INqMp-sApd0T_3LAFegFGPQes','AIzaSyBxLFuYkCyMmwOuXmKbmsWBHQgsH3a_TEU','AIzaSyCBc49VSCrcIo77BcOG0RtNi_xJgLOXtww']

def confapi(api):
    apikeysconf()
    global activekey
    activekey=api
    global resource
    resource=build("customsearch", "v1", developerKey=api_key[api], cache_discovery=False).cse()

def queryexcute(listr, cat, dcls):
    global keyindex
    global error
    global rn
    rn={}
    try:
        if dcls=="lists":
            if cat=="text":
                for n in range(len(listr)):
                    result=resource.list(q=listr[n], cx="c52a2088670cb8090").execute()
                    rn[listr[n]]=result
                return rn
            else:
                for n in range(len(listr)):
                    result=resource.list(q=listr[n], cx="c52a2088670cb8090", searchType="image").execute()
                    rn[listr[n]]=result
                return rn
        elif dcls=="dicts":
            if cat=="text":
                for n in listr:
                    rd={}
                    result=resource.list(q=listr[n], cx="c52a2088670cb8090").execute()
                    rd[n]=result
                    rn[listr[n]]=rd
                return rn
            else:
                for n in listr:
                    rd={}
                    result=resource.list(q=listr[n], cx="c52a2088670cb8090", searchType="image").execute()
                    rd[n]=result
                    rn[listr[n]]=rd
                return rn


    except:
        if activekey<len(api_key)-1:
            try:
                api=activekey+1
                confapi(api)
                rn=queryexcute(listr, cat, dcls)
                return rn
            except:
                error=1
                return error
        else:
            try:
                apic=str(input("enter api key"))
                api_key.append(apic)
                api=activekey+1
                confapi(api)
                print(api_key)
                rn=queryexcute(listr, cat, dcls)
                return rn
            except:
                error=1
                return error
    finally:
        pass



def cerequest(listr, cat, api, dcls):
    confapi(api)
    rn=queryexcute(listr, cat, dcls)
    return rn

def pass_dataimgi(imgdata):
    global imgdic
    imgdic={}
    for data in imgdata:
        imgname={}
        datu=imgdata[data]
        for dt in datu:
            limg=[]
            imgresult=datu[dt]
            if "items" in imgresult:
                for imgitem in imgresult["items"]:
                    imgpd={}
                    if "title" in imgitem:
                        imgpd["title"]=imgitem["title"]
                    else:
                        imgpd["title"]="hakunajibu"

                    if "snippet" in imgitem:
                        imgpd["snippet"]=imgitem["snippet"]
                    else:
                        imgpd["snippet"]="hakunajibu"


                    if "link" in imgitem:
                        imgpd["link"]=imgitem["link"]
                    else:
                        imgpd["link"]="hakunajibu"

                    if "pagemap" in imgitem:
                        mrf=imgitem["pagemap"]
                        if "metatags" in mrf:
                            mt=mrf["metatags"]
                            diimg=mt[0]
                        else:
                            diimg={}
                    else:
                        diimg={}

                    if "moddate" in diimg:
                        imgpd["moddate"]=diimg["moddate"]
                    else:
                        imgpd["moddate"]="hakunajibu"
                    if "creationdate" in diimg:
                        imgpd["creationdate"]=diimg["creationdate"]
                    else:
                        imgpd["creationdate"]="hakunajibu"
                    if "creator" in diimg:
                        imgpd["creator"]=diimg["creator"]
                    else:
                        imgpd["creator"]="hakunajibu"

                    if "author" in diimg:
                        imgpd["author"]=diimg["author"]
                    else:
                        imgpd["author"]="hakunajibu"
                    if "producer" in diimg:
                        imgpd["producer"]=diimg["producer"]
                    else:
                        imgpd["producer"]="hakunajibu"
                    if "appligent" in diimg:
                        imgpd["appligent"]=diimg["appligent"]
                    else:
                        imgpd["appligent"]="hakunajibu"
                    if "title" in diimg:
                        imgpd["title1"]=diimg["title"]
                    else:
                        imgpd["title1"]="hakunajibu"
                    limg.append(imgpd)
            else:
                imgpd={}
                imgpd["title"]="hakunajibu"
                imgpd["snippet"]="hakunajibu"
                imgpd["link"]="hakunajibu"
                imgpd["moddate"]="hakunajibu"
                imgpd["creationdate"]="hakunajibu"
                imgpd["creator"]="hakunajibu"
                imgpd["author"]="hakunajibu"
                imgpd["producer"]="hakunajibu"
                imgpd["appligent"]="hakunajibu"
                imgpd["title1"]="hakunajibu"
                limg.append(imgpd)
            imgname[dt]=limg
        imgdic[data]=imgname

    return imgdic

def image_resulti(imgdata):
    imgdici={}
    for data in imgdata:
        imgname={}
        datu=imgdata[data]
        for dt in datu:
            limgi=[]
            imgresulti=datu[dt]
            if "items" in imgresulti:
                for imgitemi in imgresulti["items"]:
                    imgpdi={}
                    if "title" in imgitemi:
                        imgpdi["title"]=imgitemi["title"]
                    else:
                        imgpdi["title"]="hakunajibu"

                    if "snippet" in imgitemi:
                        imgpdi["snippet"]=imgitemi["snippet"]
                    else:
                        imgpdi["snippet"]="hakunajibu"

                    linksi = imgitemi["link"]
                    if linksi.endswith(".jpg") or linksi.endswith(".png") or linksi.endswith(".webp"):
                        imgpdi["link"]=imgitemi["link"]
                        if linksi.startswith("x-raw-image:"):
                            imgpdi["link"]="hakunajibu"
                        else:
                            imgpdi["link"]=imgitemi["link"]
                    else:
                        imgpdi["link"]="hakunajibu"



                    limgi.append(imgpdi)
            else:
                imgpdi={}
                imgpdi["title"]="hakunajibu"
                imgpdi["snippet"]="hakunajibu"
                imgpdi["link"]="hakunajibu"
                limgi.append(imgpdi)
            imgname[dt]=limgi
        imgdici[data]=imgname
    return imgdici


def pass_datatext(textdata):
    global textdic
    textdic={}
    for data in textdata:
        imgresult=textdata[data]
        limg=[]
        if "items" in imgresult:
            for imgitem in imgresult["items"]:
                imgpd={}
                if "title" in imgitem:
                    imgpd["title"]=imgitem["title"]
                else:
                    imgpd["title"]="hakunajibu"

                if "snippet" in imgitem:
                    imgpd["snippet"]=imgitem["snippet"]
                else:
                    imgpd["snippet"]="hakunajibu"

                if "link" in imgitem:
                    imgpd["link"]=imgitem["link"]
                else:
                    imgpd["link"]="hakunajibu"

                if "pagemap" in imgitem:
                    mrf=imgitem["pagemap"]
                    if "metatags" in mrf:
                        mt=mrf["metatags"]
                        diimg=mt[0]
                    else:
                        diimg={}

                else:
                    diimg={}

                if "moddate" in diimg:
                    imgpd["moddate"]=diimg["moddate"]
                else:
                    imgpd["moddate"]="hakunajibu"
                if "creationdate" in diimg:
                    imgpd["creationdate"]=diimg["creationdate"]
                else:
                    imgpd["creationdate"]="hakunajibu"
                if "creator" in diimg:
                    imgpd["creator"]=diimg["creator"]
                else:
                    imgpd["creator"]="hakunajibu"

                if "author" in diimg:
                    imgpd["author"]=diimg["author"]
                else:
                    imgpd["author"]="hakunajibu"
                if "producer" in diimg:
                    imgpd["producer"]=diimg["producer"]
                else:
                    imgpd["producer"]="hakunajibu"
                if "appligent" in diimg:
                    imgpd["appligent"]=diimg["appligent"]
                else:
                    imgpd["appligent"]="hakunajibu"
                if "title" in diimg:
                    imgpd["title1"]=diimg["title"]
                else:
                    imgpd["title1"]="hakunajibu"
                limg.append(imgpd)

        else:
            imgpd={}
            imgpd["title"]="hakunajibu"
            imgpd["snippet"]="hakunajibu"
            imgpd["link"]="hakunajibu"
            imgpd["moddate"]="hakunajibu"
            imgpd["creationdate"]="hakunajibu"
            imgpd["creator"]="hakunajibu"
            imgpd["author"]="hakunajibu"
            imgpd["producer"]="hakunajibu"
            imgpd["appligent"]="hakunajibu"
            imgpd["title1"]="hakunajibu"
            limg.append(imgpd)
        textdic[data]=limg

    return textdic

def image_result(textdata1):
    imgdici1={}
    for data in textdata1:
        limgi=[]
        imgresulti=textdata1[data]
        if "items" in imgresulti:
            for imgitemi in imgresulti["items"]:
                imgpdi={}
                if "title" in imgitemi:
                    imgpdi["title"]=imgitemi["title"]
                else:
                    imgpdi["title"]="hakunajibu"
                if "title" in imgitemi:
                    imgpdi["snippet"]=imgitemi["snippet"]
                else:
                    imgpdi["snippet"]="hakunajibu"
                if "link" in imgitemi:
                    linksi = imgitemi["link"]
                else:
                    linksi = "hakunajibu"

                if linksi.endswith(".jpg") or linksi.endswith(".png") or linksi.endswith(".webp") or linksi.endswith(".jepg"):
                    imgpdi["link"]=imgitemi["link"]
                    if linksi.startswith("x-raw-image:"):
                        imgpdi["link"]="hakunajibu"
                    else:
                        imgpdi["link"]=imgitemi["link"]
                else:
                    imgpdi["link"]="hakunajibu"
                limgi.append(imgpdi)
        else:
            imgpdi={}
            imgpdi["title"]="hakunajibu"
            imgpdi["snippet"]="hakunajibu"
            imgpdi["link"]="hakunajibu"
            limgi.append(imgpdi)
        imgdici1[data]=limgi
    return imgdici1

 #plag checker plag checker
def rrr():
    tmp_name=''.join(random.choice(string.ascii_letters) for x in range(10))
    return tmp_name

def extracttext_image(saved):
    j=0
    for i in range(len(saved)):
        j=i
        try:
            img=Image.open(saved[i])
            text=tess.image_to_string(img)
            j=i
        except:
            text="hakunajibu"
            j=i
            pass

    os.remove(saved[j])
    return text
def imgtxtff(c):
    jj=[]
    for i in range (len(c)):
        try:
            response = requests.get(c[i])
            jk=c[i]
            ext=jk[-3:]
            name="/home/xz0r/upload/imgkk/%s.%s"%(rrr(),ext)
            file = open(name, "wb")
            file.write(response.content)
            jj.append(name)
            file.close()
            kll=extracttext_image(jj)
        except:
            kll="hakunajibu"
    return kll

def imgtextextractmlist(searchimg):
    for key in searchimg:
        ls=searchimg[key]
        for i in range(len(ls)):
            lk=[2]
            copy=ls[i]
            if copy["link"]!="hakunajibu":
                url =copy["link"]
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
                try:
                    lk[0]=url
                    text=imgtxtff(lk)
                    text=splitfstop1(text)
                    text=" ".join(text)
                    text=text.split(".")
                    text=" ".join(text)
                    text=process_list(text)

                except:
                    copy["added"]="hakunajibu"
                    text=""
                if text!="":
                    copy["added"]=text
                else:
                    copy["added"]="hakunajibu"
                ls[i]=copy


            else:
                copy["added"]="hakunajibu"
                ls[i]=copy
        searchimg[key]=ls
    return searchimg

def stop_wordrm(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    text = " ".join(filtered_sentence)
    return text

def stemlem(text):
    listr=[]
    lancaster=LancasterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    sentence = text
    punctuations="?:!.,;"
    sentence_words = nltk.word_tokenize(sentence)
    for word in sentence_words:
        if word in punctuations:
            sentence_words.remove(word)

    for word in sentence_words:
        word=lancaster.stem(word)
        listr.append(wordnet_lemmatizer.lemmatize(word))

    listr=" ".join(listr)
    return listr


def tocenizer(wordls):
    dicre={}
    for key in wordls:
        d=wordls[key]
        for i in range(len(d)):
            dicob=d[i]
            dicob["snippettk"]=process_list(dicob["snippet"])
            dicob["snippettk"]=stop_wordrm(dicob["snippettk"])
            dicob["snippettk"]=stemlem(dicob["snippettk"])
            dicob["keytk"]=stop_wordrm(key)
            dicob["keytk"]=stemlem(dicob["keytk"])
            d[i]=dicob
        dicre[key]=d
    return dicre

def dense_contrib(ctext, totalw):
    tkcontr={}
    for x in ctext:
        tkd=ctext[x]
        for c in range(len(tkd)):
            dic=tkd[c]
            dic["datadense"]=(dic["wordlen"]/totalw)*100
            dic["totalw"]=totalw
            tkd.append(dic)

        tkcontr[x]=tkd
    return tkcontr



def text_dense(text):
    totalw=0
    ctext={}
    for key in text:
        d=text[key]
        dss=d[0]
        tokee=dss["keytk"]
        tkl=tokee.count(" ")+1
        totalw=totalw +tkl
        for i in range(len(d)):
            ds=d[i]
            ds["wordlen"]=tkl
            d.append(ds)
        ctext[key]=d
    dataorg=dense_contrib(ctext, totalw)
    return dataorg

def  similaritychecker(textorg, textpross):

    txto =[textorg[0], textorg[1]]
    txtp=[textpross[0], textpross[1]]

    vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
    similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

    vectors = vectorize(txtp)

    s_vectors = list(zip(txto, vectors))
    plagiarism_results = []
    for text_a, text_vector_a in s_vectors:
        new_vectors =s_vectors.copy()
        current_index = new_vectors.index((text_a, text_vector_a))
        del new_vectors[current_index]
        for text_b , text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            plagiarism_results.append(sim_score)
    return plagiarism_results
def plagqual(listr):
    listjj=[]

    a=nltk.word_tokenize(listr[0])
    b=nltk.word_tokenize(listr[1])
    for i in range(len(a)):
        if a[i] in b:
            listjj.append(a[i])
        else:
            pass
    ssr=len(listjj)
    tss=len(a)
    ps=(ssr/tss)*100
    bb=int(ps)
    if bb!=0:
        c=ps%bb
    else:
        c=0
    if c>=0.5:
        if (bb % 2) == 0:
            eve=True
        else:
            eve=False
        if eve:
            ps=round(ps+1)-1
        else:
            ps=round(ps)
    else:
        pass
    return ps

def totalplag(text):
    sum0=0
    sum2=0
    lsrr=[]
    for key in text:
        lskey=text[key]
        for j in range(len(lskey)):
            dt=lskey[j]
            a=dt["persplagd"]
            b=int(a)
            if b!=0:
                c=a%b
            else:
                c=0
            if c>=0.5:
                if (b % 2) == 0:
                    eve=True
                else:
                    eve=False
                if eve:
                    sum2=sum2+(round(dt["persplagd"]+1)-1)
                else:
                    sum2=sum2+round(dt["persplagd"])
            else:
                sum2=sum2+dt["persplagd"]

            sum0=sum0+dt["persplagd"]
    if sum2>100:
        lsrr.append(round(sum0,2))
    else:
        lsrr.append(round(sum2,2))
    lsrr.append(text)
    return lsrr

def  check_plagiarism(dps):
    dics={}
    listrr=[]
    exl=len(dps.keys())
    sumorg=0
    lnn=0
    ite=0
    for ds in dps:
        datals=dps[ds]
        maxp=0.0
        for i in range(len(datals)):
            lis1=[]
            lis2=[]
            dics2=datals[i]
            txp1=dics2["keytk"]
            txp2=dics2["snippettk"]
            lis2.append(txp1)
            lis2.append(txp2)
            texto1=ds
            texto2=dics2["snippet"]
            lis1.append(texto1)
            lis1.append(texto2)
            #pers=similaritychecker(lis1, lis2)
            pers=plagqual(lis2)
            dics2["persplag"]=pers
            pss=(dics2["persplag"]/100)*dics2["datadense"]
            bb=int(pss)
            if bb!=0:
                c=pss%bb
            else:
                c=0
            if c>=0.5:
                if (bb % 2) == 0:
                    eve=True
                else:
                    eve=False
                if eve:
                    pss=round(pss+1)-1
                else:
                    pss=round(pss)
            else:
                pass
            dics2["persplagd"]=pss
            datals.append(dics2)
            if maxp<pers:
                maxp=pers
                listrr=[datals[i]]
                ite=i

            else:
                pass
        lnn=lnn+1
        sumorg=dics2["persplagd"]+sumorg
        if lnn==exl-1:
            if sumorg>100:
                s2=sumorg-dics2["persplagd"]
                dt=100-s2
                dics2["persplagd"]=dt
                datals[ite]=dics2
                listrr=[datals[ite]]
            else:
                pass
        else:
            pass

        dics[ds]=listrr
    dics=totalplag(dics)
    return dics



def tocenizerimg(wordls):
    dicre={}
    for key in wordls:
        d=wordls[key]
        for i in range(len(d)):
            dicob=d[i]
            dicob["snippettk"]=process_list(dicob["added"])
            dicob["snippettk"]=stop_wordrm(dicob["snippettk"])
            dicob["snippettk"]=stemlem(dicob["snippettk"])
            dicob["keytk"]=stop_wordrm(key)
            dicob["keytk"]=stemlem(dicob["keytk"])
            d[i]=dicob
        dicre[key]=d
    return dicre
def plagreport(txt, pdfinfo):
    txt=tocenizer(txt)
    pse=text_dense(txt)
    txt=check_plagiarism(pse)
    plagrdict={}
    plagtdict={}
    plagtotal={}

    plag=txt[0]
    detail=txt[1]
    if pdfinfo["CreationDate"]!="hakunajibu":
        crdt=pdfinfo["CreationDate"]
        if crdt.startswith("D"):
            crdt=crdt[2:]
        else:
            crdt=crdt
        cryy=str(crdt[0:4])
        crmm=str(crdt[4:6])
        crdd=str(crdt[6:8])
        crhr=str(crdt[8:10])
        crmn=str(crdt[10:12])
        crsec=str(crdt[12:])
        chrr=" HR-"
        ccrd="creation date:-"
        csp=":"
        csp2="-"
    else:
        chrr=""
        ccrd=""
        csp=":"
        csp2=""
        cryy=""
        crmm=""
        crdd=""
        crhr=""
        crmn=""
        crsec=""
    if pdfinfo["Creator"]!="hakunajibu":
        pdcrt=str(pdfinfo["Creator"])
    else:
        pdcrt=""
    if pdfinfo["Producer"]!="hakunajibu":
        pdcrtb=str(pdfinfo["Producer"])
    else:
        pdcrtb=""
    cdate=cryy+csp2+crmm+csp2+crdd+chrr+crhr+csp+crmn+csp+crsec
    producer=pdcrtb
    creator=pdcrt
    plagrdict["cdate"]=cdate
    plagrdict["producer"]=producer
    plagrdict["creator"]=creator
    subtextdic={}
    keytextdic={}


    for key in detail:
        subtextdic={}
        klm=detail[key]
        if klm!=[]:
            data=klm[0]
        else:
            data={}
            data['persplagd']=0.0
            data['creationdate']="hakunajibu"
            data['creator']="hakunajibu"
            data['author']="hakunajibu"
            data['producer']="hakunajibu"
            data['title']=""
            data['snippet']=""
            data['link']=""
        #data=klm[0]
        aa=round(float(data['persplagd']), 1)
        if data['creationdate']!="hakunajibu":
            date=data['creationdate'].split(":")
            date=date[1]
            hrr=" HR-"
            crd="<b>creation date:-</b>"
            sp=":"
            sp2="-"
            yy=str(date[0:4])
            mm=str(date[4:6])
            dd=str(date[6:8])
            hr=str(date[8:10])
            mn=str(date[10:12])
            sec=str(date[12:])
        else:
            sp=""
            sp2=""
            hrr=""
            crd=""
            date=""
            yy=""
            mm=""
            dd=""
            hr=""
            mn=""
            sec=""

        if data['creator']!="hakunajibu":
            crt=str(data['creator'])
        else:
            crt=""
        if data['author']!="hakunajibu":
            crtb=str(data['author'])
        else:
            crtb=""
        if data['producer']!="hakunajibu":
            crtc=str(data['producer'])
        else:
            crtc=""
        subtextdic["cdate"]=yy+sp2+mm+sp2+dd+hrr+hr+sp +mn+sp+sec
        subtextdic["author"]=crtb
        subtextdic["creator"]=crt
        subtextdic["producer"]=crtc
        subtextdic["title"]=data['title']
        subtextdic["snippet"]=data['snippet']
        subtextdic["link"]=data['link']
        subtextdic["plagp"]=str(aa)
        keytextdic[str(key)]=subtextdic
    plag=float(plag)
    if plag >100:
        plag=100.00
    else:
        pass
    plagtotal["totalp"]=str(plag)
    plagtotal["meta"]=plagrdict
    plagtotal["text"]=keytextdic
    return plagtotal


def plagreporti(txt):
    procesed2=tocenizerimg(txt)
    pse2=text_dense(procesed2)
    rsult2=check_plagiarism(pse2)
    plpp=rsult2[0]
    detailimg=rsult2[1]
    kkk=""
    kkk=kkk+"<div class='h5 text-center'>SUSPECTED IMAGE WHERE TEXT EXTRACTED! REVIEW IT MANUALLY</div>"
    for key in detailimg:
        lsimgt=detailimg[key]
        for i in range(len(lsimgt)):
            dtt=lsimgt[i]
            plg=round(float(dtt['persplagd']), 1)
            plagia='<b>text: '+key+'</b>'
            link=dtt['link']
            info='<p class="card-text h4">'+plagia+'</p><i class="h3">suspected plagiarism </i><button class="btn btn-warning">'+str(plg)+'%</button>'
            info2='<b><p class="h3">source:</b><a href="' + link + '">"'+link+'"</a></p><i>'
            kkk=kkk+'<div class="col-sm-12"> <table class="table"><th><b>suspected source: </b><img class="col-sm-12" style="width:100%" src="'+link+'">'+info+info2+'<th></table></div>'

    kkk=kkk+"<div class='h1 text-center col-md-12 btn btn-info'>SUSPECTED PLAGIARISM AMOUNT</div>"+""+"<div class='col-md-12 h1 text-center btn btn-danger'>"+str(plpp)+"%</div>"
    kkk=display(HTML(kkk))
    return kkk

def imgtextextractmlist1(txt):
    for key in txt:
        imgdata=imgtextextractmlist(txt[key])
        txt[key]=imgdata
    return txt
def tokendic(text):
    dicre={}
    dicre2={}
    for key1 in text:
        wordls=text[key1]
        for key in wordls:
            d=wordls[key]
            dicre={}
            for i in range(len(d)):
                dicob=d[i]
                dicob["snippettk"]=process_list(dicob["added"])
                dicob["snippettk"]=stop_wordrm(dicob["snippettk"])
                dicob["snippettk"]=stemlem(dicob["snippettk"])
                dicob["keytk"]=stop_wordrm(key1)
                dicob["keytk"]=stemlem(dicob["keytk"])
                d[i]=dicob
            dicre[key]=d
        dicre2[key1]=dicre
    return dicre2
def text_denseimg(txt):
    for key in txt:
        answeer=text_dense(txt[key])
        txt[key]=answeer
    return txt
def  check_plagiarismimg(txt):
    for key in txt:
        answeer=check_plagiarism(txt[key])
        txt[key]=answeer
    return txt

def plagreportimg(txt):
    dkk=tokendic(txt)
    persw=text_denseimg(dkk)
    txt=check_plagiarismimg(persw)
    kkk=""
    kkk=kkk+"<div class='h5 text-center'>DOCUMENT IMAGE REPORT</div>"
    for keys in txt:
        dataf=txt[keys]
        plpp=dataf[0]
        detailimg=dataf[1]
        for key in detailimg:
            lsimgt=detailimg[key]
            for i in range(len(lsimgt)):
                dtt=lsimgt[i]
                plg=round(float(dtt['persplagd']), 1)
                plagia='<b>document image: <img src="'+key+'"></b>'
                link=dtt['link']
                info='<p class="card-text h4">'+plagia+'</p><i class="h3">suspected plagiarism </i><button class="btn btn-warning">'+str(plg)+'%</button>'
                info2='<b><p class="h3">source:</b><a href="' + link + '">"'+link+'"</a></p><i>'
                kkk=kkk+'<div class="col-sm-12"> <table class="table"><th><b>suspected image:</b><img class="col-sm-12" style="width:100%" src="'+link+'">'+info+info2+'<th></table></div>'

        kkk=kkk+"<div class='h1 text-center col-md-12 btn btn-info'>SUSPECTED PLAGIARISM</div>"+""+"<div class='col-md-12 h1 text-center btn btn-danger'>"+str(plpp)+"%</div>"
    kkk=display(HTML(kkk))
    return kkk
def tokendic2(text):
    dicre={}
    dicre2={}
    for key1 in text:
        wordls=text[key1]
        for key in wordls:
            d=wordls[key]
            dicre={}
            for i in range(len(d)):
                dicob=d[i]
                dicob["snippettk"]=process_list(dicob["snippet"])
                dicob["snippettk"]=stop_wordrm(dicob["snippettk"])
                dicob["snippettk"]=stemlem(dicob["snippettk"])
                dicob["keytk"]=stop_wordrm(key1)
                dicob["keytk"]=stemlem(dicob["keytk"])
                d[i]=dicob
            dicre[key]=d
        dicre2[key1]=dicre
    return dicre2
def plagreportimgi(txt):
    dkk=tokendic2(txt)
    persw=text_denseimg(dkk)
    txt=check_plagiarismimg(persw)
    kkk=""
    kkk=kkk+"<div class='h5 text-center'>PLAGIARIZED TEXT FROM IMAGE EMBEDED ON DOCUMENT</div>"
    for keys in txt:
        dataf=txt[keys]
        plpp=dataf[0]
        detailimg=dataf[1]
        for key in detailimg:
            lsimgt=detailimg[key]
            for i in range(len(lsimgt)):
                dtt=lsimgt[i]
                plg=round(float(dtt['persplagd']), 1)
                plagia='<h3>'+dtt["snippet"]+'</h3><br><i>text snippet:'+dtt["snippet"]+'</i>'
                link=dtt['link']
                info='<p class="card-text h4">'+plagia+'</p><i class="h3">suspected plagiarism </i><button class="btn btn-warning">'+str(plg)+'%</button>'
                info2='<b><p class="h3">source:</b><a href="' + link + '">"'+link+'"</a></p><i>'
                kkk=kkk+'<div class="col-sm-12">Document image <table class="table"><th> <img class="col-sm-12" style="width:100%" src="'+key+'">'+info+info2+'<th></table></div>'

        kkk=kkk+"<div class='h1 text-center col-md-12 btn btn-info'>SUSPECTED PLAGIARISM</div>"+""+"<div class='col-md-12 h1 text-center btn btn-danger'>"+str(plpp)+"%</div>"
    kkk=display(HTML(kkk))
    return kkk
def plaggg(path, ftype):
    global student_files
    global student_notes
    global vectorize
    global similarity
    global vectors
    global s_vectors
    global plagiarism_results
    global pathe
    pathe=0
    path=str(path)
    if path.endswith("/"):
        pass
    else:
        path=path+"/"
    try:
        if ftype=="pdf":
            student_files = [doc for doc in os.listdir(path) if doc.endswith('.'+ftype)]
            student_notes =[" ".join(stream(path+File)) for File in  student_files]
            vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
            similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])
            vectors = vectorize(student_notes)
            s_vectors = list(zip(student_files, vectors))
            plagiarism_results = set()
        elif ftype=="doc":
            student_files = [doc for doc in os.listdir(path) if doc.endswith('.'+ftype)]
            student_notes =[" ".join(stream(path+File)) for File in  student_files]
            vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
            similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])
            vectors = vectorize(student_notes)
            s_vectors = list(zip(student_files, vectors))
            plagiarism_results = set()

        else:
            student_files = [doc for doc in os.listdir(path) if doc.endswith('.'+ftype)]
            student_notes =[open(path+File).read() for File in  student_files]
            vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
            similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])
            vectors = vectorize(student_notes)
            s_vectors = list(zip(student_files, vectors))
            plagiarism_results = set()
    except:
        pathe=1

def check_plagiarismi():
    if pathe==0:
        for student_a, text_vector_a in s_vectors:
            new_vectors =s_vectors.copy()
            current_index = new_vectors.index((student_a, text_vector_a))
            del new_vectors[current_index]
            for student_b , text_vector_b in new_vectors:
                sim_score = similarity(text_vector_a, text_vector_b)[0][1]
                sim_score=sim_score*100
                sim_score=round(float(sim_score), 2)
                sim_score=str(sim_score)+"%"
                student_pair = sorted((student_a, student_b))
                score = (student_pair[0], student_pair[1],sim_score)
                plagiarism_results.add(score)
        return plagiarism_results
    else:
        print("some error occured4")

def mutualfileplag(path, ftype):
    plaggg(path, ftype)
    try:
        dictiii={}
        for data in check_plagiarismi():
            listiix=[]
            axx=list(data)
            listiix.append(axx[1])
            listiix.append(axx[2])
            dictiii[axx[0]]=listiix      
        print(dictiii)
    except:
        print("no data")
    return dictiii
