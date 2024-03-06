import bcrypt

# https://stackoverflow.com/a/23768422/19123103
def get_hashed_password(plain_text_password, encoding="utf-8"):
    # Hash a password for the first time
    return bcrypt.hashpw(plain_text_password.encode(encoding), bcrypt.gensalt(rounds=12))

def check_password(plain_text_password, hashed_password, encoding="utf-8"):
    # Check hashed password. Using bcrypt, the salt is saved into the hash itself
    return bcrypt.checkpw(plain_text_password.encode(encoding), hashed_password)