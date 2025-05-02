import google.generativeai as genai
import os
import sys
from colorama import init, Fore, Style

# Colorama'yı başlat
init()

def clear_screen():
    """Ekranı temizle"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_colored(text, color=Fore.WHITE, style=Style.NORMAL):
    """Renkli metin yazdır"""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def load_api_key():
    """API anahtarını dosyadan yükle"""
    try:
        with open("gemini_api_key.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def save_api_key(api_key):
    """API anahtarını dosyaya kaydet"""
    with open("gemini_api_key.txt", "w") as f:
        f.write(api_key)

def setup_gemini():
    """Gemini API'yi yapılandır"""
    api_key = load_api_key()
    
    if not api_key:
        print_colored("Gemini API anahtarı bulunamadı.", Fore.YELLOW)
        api_key = input("Lütfen Gemini API anahtarınızı girin: ").strip()
        
        if api_key:
            try:
                # API anahtarını test et
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content("Test connection")
                
                # API anahtarını kaydet
                save_api_key(api_key)
                print_colored("API anahtarı başarıyla kaydedildi!", Fore.GREEN)
                return model
            except Exception as e:
                print_colored(f"API anahtarı geçersiz: {str(e)}", Fore.RED)
                return None
        else:
            print_colored("API anahtarı girilmedi.", Fore.RED)
            return None
    else:
        try:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('v1beta')
        except Exception as e:
            print_colored(f"API yapılandırma hatası: {str(e)}", Fore.RED)
            return None

def chat_loop(model):
    """Ana sohbet döngüsü"""
    chat = model.start_chat(history=[])
    
    print_colored("\nGemini Chat'e hoş geldiniz!", Fore.CYAN, Style.BRIGHT)
    print_colored("Çıkmak için 'quit' veya 'exit' yazın.\n", Fore.YELLOW)
    
    while True:
        try:
            # Kullanıcı girdisi
            user_input = input(f"{Fore.GREEN}Sen: {Style.RESET_ALL}")
            
            # Çıkış kontrolü
            if user_input.lower() in ['quit', 'exit']:
                print_colored("\nGörüşmek üzere!", Fore.CYAN)
                break
            
            # Boş girdi kontrolü
            if not user_input.strip():
                continue
            
            # Gemini'den yanıt al
            response = chat.send_message(user_input)
            
            # Yanıtı göster
            print(f"\n{Fore.BLUE}Gemini: {Style.RESET_ALL}{response.text}\n")
            
        except KeyboardInterrupt:
            print_colored("\n\nGörüşmek üzere!", Fore.CYAN)
            break
        except Exception as e:
            print_colored(f"\nBir hata oluştu: {str(e)}", Fore.RED)
            print_colored("Yeni bir sohbet başlatılıyor...\n", Fore.YELLOW)
            chat = model.start_chat(history=[])

def main():
    """Ana program"""
    clear_screen()
    
    # Gemini modelini yapılandır
    model = setup_gemini()
    
    if model:
        try:
            chat_loop(model)
        except Exception as e:
            print_colored(f"\nBeklenmeyen bir hata oluştu: {str(e)}", Fore.RED)
    else:
        print_colored("\nProgram başlatılamadı. Lütfen geçerli bir API anahtarı girin.", Fore.RED)

if __name__ == "__main__":
    main() 