from gtts import gTTS

import pyttsx3
import os

def  diger(ses):
# Türkçe metin


    # gTTS ile Türkçe metni sese çevir
    tts = gTTS(text=ses, lang='tr', slow=False)

    # Sesi bir dosyaya kaydet
    dosya_adi = "ses.mp3"
    tts.save(dosya_adi)

    # Kaydedilen sesi oynat (Windows için)
    os.system(f"start {dosya_adi}")



def ses(metin):

    # Motoru başlat
    motor = pyttsx3.init()

    # Tüm sesleri listele
    sesler = motor.getProperty('voices')

    # Konuşma hızını ve ses seviyesini ayarla
    motor.setProperty('rate', 150)  # Konuşma hızı
    motor.setProperty('volume', 10)  # Ses seviyesi

    # Metni oku
    motor.say(metin)
    motor.runAndWait()


metin = """
 Ey Türk gençliği! Birinci vazifen; Türk istiklalini, Türk cumhuriyetini, ilelebet muhafaza ve müdafaa etmektir.

   Mevcudiyetinin ve istikbalinin yegâne temeli budur. Bu temel, senin en kıymetli hazinendir. İstikbalde dahi seni bu hazineden mahrum etmek isteyecek dâhilî ve haricî bedhahların olacaktır. Bir gün, istiklal ve cumhuriyeti müdafaa mecburiyetine düşersen, vazifeye atılmak için içinde bulunacağın vaziyetin imkân ve şeraitini düşünmeyeceksin. Bu imkân ve şerait, çok namüsait bir mahiyette tezahür edebilir. İstiklal ve cumhuriyetine kastedecek düşmanlar, bütün dünyada emsali görülmemiş bir galibiyetin mümessili olabilirler. Cebren ve hile ile aziz vatanın bütün kaleleri zapt edilmiş, bütün tersanelerine girilmiş, bütün orduları dağıtılmış ve memleketin her köşesi bilfiil işgal edilmiş olabilir. Bütün bu şeraitten daha elim ve daha vahim olmak üzere, memleketin dâhilinde iktidara sahip olanlar, gaflet ve dalalet ve hatta hıyanet içinde bulunabilirler. Hatta bu iktidar sahipleri, şahsi menfaatlerini müstevlilerin siyasi emelleriyle tevhit edebilirler. Millet, fakruzaruret içinde harap ve bitap düşmüş olabilir.

   Ey Türk istikbalinin evladı! İşte, bu ahval ve şerait içinde dahi vazifen, Türk istiklal ve cumhuriyetini kurtarmaktır. Muhtaç olduğun kudret, damarlarındaki asil kanda mevcuttur.
"""
