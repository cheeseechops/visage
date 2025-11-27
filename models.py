from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# Junction table for many-to-many relationship between images and people
image_people = db.Table('image_people',
    db.Column('image_id', db.Integer, db.ForeignKey('images.id'), primary_key=True),
    db.Column('person_id', db.Integer, db.ForeignKey('people.id'), primary_key=True)
)

# Junction table for many-to-many relationship between videos and people
video_people = db.Table('video_people',
    db.Column('video_id', db.Integer, db.ForeignKey('videos.id'), primary_key=True),
    db.Column('person_id', db.Integer, db.ForeignKey('people.id'), primary_key=True)
)

class Person(db.Model):
    __tablename__ = 'people'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    thumbnail_path = db.Column(db.String(255))
    is_confirmed = db.Column(db.Boolean, default=False)
    is_favorite = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Many-to-many relationship with images
    images = db.relationship('Image', secondary=image_people, backref='people')
    
    @property
    def image_count(self):
        return len(self.images)
    
    def to_dict(self):
        # Always use the explicit thumbnail_path if set (centered profile picture)
        if self.thumbnail_path:
            cover_url = self.thumbnail_path
        else:
            cover_url = f'https://via.placeholder.com/150x200/6B7280/FFFFFF?text={self.name[:5]}'
        return {
            'id': self.id,
            'name': self.name,
            'thumbnail': cover_url,
            'thumbnail_path': self.thumbnail_path,  # for explicit use in templates
            'image_count': self.image_count,
            'is_favorite': self.is_favorite,
            'is_confirmed': self.is_confirmed
        }

class Image(db.Model):
    __tablename__ = 'images'
    
    id = db.Column(db.Integer, primary_key=True)
    original_path = db.Column(db.String(255), nullable=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    hash_value = db.Column(db.String(255), unique=True)
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    is_cropped = db.Column(db.Boolean, default=False)
    is_favorite = db.Column(db.Boolean, default=False)
    tags = db.Column(db.Text)  # JSON string of tags
    recognition_suggestions = db.Column(db.Text)  # JSON string of suggestions
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        import json
        # Prefer original_path if it exists and is accessible via /static/
        image_url = f'/static/uploads/{self.filename}'
        if self.original_path and self.original_path.startswith('/static/'):
            # Skip file existence check for performance - assume it exists
            image_url = self.original_path
        return {
            'id': self.id,
            'filename': self.filename,
            'original_path': self.original_path,
            'people': [person.name for person in self.people],
            'image_url': image_url,
            'original_screenshot': self.original_path,
            'width': self.width,
            'height': self.height,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_favorite': self.is_favorite,
            'tags': json.loads(self.tags) if self.tags else [],
            'recognition_suggestions': json.loads(self.recognition_suggestions) if self.recognition_suggestions else []
        } 

class Video(db.Model):
    __tablename__ = 'videos'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    file_path = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)  # Size in bytes
    duration = db.Column(db.Float)  # Duration in seconds
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    mime_type = db.Column(db.String(100))
    thumbnail_path = db.Column(db.String(255))  # Optional thumbnail
    tags = db.Column(db.Text)  # JSON string for tags
    is_favorite = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Many-to-many relationship with people
    people = db.relationship('Person', secondary=video_people, backref='videos')
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'duration': self.duration,
            'width': self.width,
            'height': self.height,
            'mime_type': self.mime_type,
            'thumbnail_path': self.thumbnail_path,
            'video_url': self.file_path,  # For easy access in templates
            'is_favorite': self.is_favorite,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'formatted_size': self.format_file_size(),
            'formatted_duration': self.format_duration(),
            'people': [{'id': person.id, 'name': person.name} for person in self.people],
            'parsed_tags': self.parse_tags()
        }
    
    def parse_tags(self):
        """Parse JSON tags safely"""
        import json
        if self.tags:
            try:
                return json.loads(self.tags)
            except (json.JSONDecodeError, TypeError):
                return []
        return []
    
    def format_file_size(self):
        """Format file size in human readable format"""
        if not self.file_size:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if self.file_size < 1024.0:
                return f"{self.file_size:.1f} {unit}"
            self.file_size /= 1024.0
        return f"{self.file_size:.1f} TB"
    
    def format_duration(self):
        """Format duration in MM:SS or HH:MM:SS format"""
        if not self.duration:
            return "Unknown"
        
        hours = int(self.duration // 3600)
        minutes = int((self.duration % 3600) // 60)
        seconds = int(self.duration % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

# Database indexes for performance optimization
class ImageIndexes:
    """Database indexes for Image table performance optimization"""
    
    # Index for sorting by ID (most common query)
    id_desc_index = db.Index('idx_images_id_desc', 'id', postgresql_ops={'id': 'DESC'})
    
    # Index for filtering by people (unnamed images)
    people_count_index = db.Index('idx_images_people_count', 'id')
    
    # Index for favorites filtering
    is_favorite_index = db.Index('idx_images_is_favorite', 'is_favorite', 'id')
    
    # Index for created_at sorting
    created_at_index = db.Index('idx_images_created_at', 'created_at', 'id')
    
    # Index for filename lookups
    filename_index = db.Index('idx_images_filename', 'filename')
    
    # Composite index for common queries
    common_query_index = db.Index('idx_images_common_query', 'is_favorite', 'created_at', 'id')

class PersonIndexes:
    """Database indexes for Person table performance optimization"""
    
    # Index for name lookups
    name_index = db.Index('idx_people_name', 'name')
    
    # Index for confirmed status
    is_confirmed_index = db.Index('idx_people_is_confirmed', 'is_confirmed')
    
    # Index for favorites
    is_favorite_index = db.Index('idx_people_is_favorite', 'is_favorite')

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('images.id'), nullable=False)
    personality_description = db.Column(db.Text, nullable=False)
    system_prompt = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with Image
    image = db.relationship('Image', backref='chat_sessions')
    
    # Relationship with ChatMessage
    messages = db.relationship('ChatMessage', backref='session', lazy='dynamic', cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'image_id': self.image_id,
            'personality_description': self.personality_description,
            'system_prompt': self.system_prompt,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'message_count': self.messages.count()
        }

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None
        } 