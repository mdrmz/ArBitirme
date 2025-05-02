import requests
import json
import os
import sys
from colorama import init, Fore, Style

# Colorama'yı başlat
init()

# LM Studio API endpoint'i
API_URL = "http://10.52.15.98:40/v1/chat/completions"

def clear_screen():
    """Ekranı temizle"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_colored(text, color=Fore.WHITE, style=Style.NORMAL):
    """Renkli metin yazdır"""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def get_available_models():
    """Kullanılabilir modelleri listele"""
    try:
        response = requests.get("http://10.52.15.98:40/v1/models")
        if response.status_code == 200:
            models = response.json()
            return models.get('data', [])
        return []
    except Exception as e:
        print_colored(f"Model listesi alınamadı: {str(e)}", Fore.RED)
        return []

def select_model():
    """Kullanıcıdan model seçmesini iste"""
    models = get_available_models()
    
    if not models:
        print_colored("Hiç model bulunamadı!", Fore.RED)
        return None
    
    print_colored("\nKullanılabilir Modeller:", Fore.CYAN)
    for i, model in enumerate(models, 1):
        print(f"{i}. {model.get('id', 'Bilinmeyen Model')}")
    
    while True:
        try:
            choice = int(input("\nLütfen bir model seçin (numara): "))
            if 1 <= choice <= len(models):
                return models[choice-1]['id']
            else:
                print_colored("Geçersiz seçim! Lütfen tekrar deneyin.", Fore.RED)
        except ValueError:
            print_colored("Lütfen geçerli bir numara girin!", Fore.RED)

def chat_with_model(model_id):
    """Seçilen model ile sohbet et"""
    headers = {
        "Content-Type": "application/json"
    }
    
    messages = []
    
    print_colored("\nLM Studio Chat'e hoş geldiniz!", Fore.CYAN, Style.BRIGHT)
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
            
            # Mesajı sohbet geçmişine ekle
            messages.append({"role": "user", "content": user_input})
            
            # API isteği için veri hazırla
            data = {
                "model": model_id,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # API'ye istek gönder
            response = requests.post(API_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                response_data = response.json()
                assistant_message = response_data['choices'][0]['message']['content']
                
                # Asistan yanıtını sohbet geçmişine ekle
                messages.append({"role": "assistant", "content": assistant_message})
                
                # Yanıtı göster
                print(f"\n{Fore.BLUE}Model: {Style.RESET_ALL}{assistant_message}\n")
            else:
                print_colored(f"\nHata: {response.status_code} - {response.text}", Fore.RED)
            
        except KeyboardInterrupt:
            print_colored("\n\nGörüşmek üzere!", Fore.CYAN)
            break
        except Exception as e:
            print_colored(f"\nBir hata oluştu: {str(e)}", Fore.RED)
            print_colored("Yeni bir sohbet başlatılıyor...\n", Fore.YELLOW)
            messages = []

def main():
    """Ana program"""
    clear_screen()
    
    # Model seç
    model_id = select_model()
    
    if model_id:
        try:
            chat_with_model(model_id)
        except Exception as e:
            print_colored(f"\nBeklenmeyen bir hata oluştu: {str(e)}", Fore.RED)
    else:
        print_colored("\nProgram başlatılamadı. Lütfen LM Studio'nun çalıştığından emin olun.", Fore.RED)

if __name__ == "__main__":
    main() 