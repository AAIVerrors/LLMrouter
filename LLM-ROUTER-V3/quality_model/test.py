from quality_model import P2LPredictor

quality_model = P2LPredictor()
coefs = quality_model.get_coefficients('''who are you?''')
print(coefs)