import speech_recognition as sr


def ses_to_metin() -> str:
    recognizer = sr.Recognizer()
    metin_toplama = ""  # Tüm metni birleştireceğimiz değişken

    with sr.Microphone() as source:
        print("Lütfen Türkçe konuşun... (bitti diyene kadar devam edeceğim)")

        recognizer.adjust_for_ambient_noise(source)  # Ortamdaki gürültüyü algıla

        while True:
            try:
                audio = recognizer.listen(source, timeout=10)  # 10 saniye boyunca dinle
                # Google'ın Türkçe tanıma hizmetini kullan
                text = recognizer.recognize_google(audio, language='tr-TR')  # Türkçe dil desteği
                print("Algılanan Metin:", text)
                metin_toplama += text + " "  # Algılanan metni birleştir

                # Eğer "bitti" kelimesi algılanırsa döngü sonlanır
                if "bitti" in text.lower():
                    print("Bitti dediniz, işlem sonlanıyor.")
                    break

            except sr.UnknownValueError:
                print("Ses anlaşılamadı.")
            except sr.RequestError as e:
                print(f"Google Ses Tanıma hizmeti çalıştırılamadı: {e}")
            except Exception as e:
                print(f"Hata oluştu: {e}")

    return metin_toplama


# Fonksiyonu çalıştır
if __name__ == "__main__":
    metin = ses_to_metin()
    print("\nToplanan Tüm Metin:")
    print(metin)  # Sonunda tüm metni yazdırıyoruz
