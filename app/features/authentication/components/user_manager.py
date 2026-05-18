def get_user_by_id(user_id: int, db, User):
    return db.query(User).filter(User.id == user_id).first()