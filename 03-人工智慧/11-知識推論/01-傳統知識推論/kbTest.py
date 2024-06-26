from kb import KB

code = "A<=B. B<=C&D. C<=E. D<=F. E. F. Z<=C&D&G."
kb1 = KB()
kb1.load(code)
kb1.forwardChaining()
