import json
import redis
from config import settings
import structlog
import hashlib

logger = structlog.get_logger(__name__)

class RedisCache:
    """Simple cache using Redis for semantic/exact match queries."""
    
    def __init__(self):
        self.redis_client = None
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("redis_connected", url=settings.REDIS_URL)
        except Exception as e:
            logger.warning("redis_connection_failed", error=str(e))
            self.redis_client = None

    def _get_key(self, query: str) -> str:
        # Use a hash of the query for simpler key storage
        query_hash = hashlib.md5(query.lower().strip().encode('utf-8')).hexdigest()
        return f"healrag_cache:{query_hash}"

    def get_cached_response(self, query: str) -> dict:
        if not self.redis_client:
            return None
            
        key = self._get_key(query)
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                logger.info("cache_hit", query=query)
                return json.loads(cached_data)
        except Exception as e:
            logger.error("redis_read_error", error=str(e))
        return None

    def set_cached_response(self, query: str, response: dict, ttl_seconds: int = 3600):
        if not self.redis_client:
            return
            
        key = self._get_key(query)
        try:
            # We don't cache failed/errored responses
            if response.get("status") == "success":
                self.redis_client.setex(key, ttl_seconds, json.dumps(response))
                logger.info("cache_set", query=query)
        except Exception as e:
            logger.error("redis_write_error", error=str(e))

redis_cache = RedisCache()
