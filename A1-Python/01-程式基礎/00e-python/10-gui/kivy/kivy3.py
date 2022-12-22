from kivy.app import App
from kivy.uix.button import Button

class Mybutton(Button):
    text="Click me!"
    on_press =lambda a : print("My Button")    
    
class TutorialApp(App):
    def build(self):
        return Mybutton()
TutorialApp().run()
 