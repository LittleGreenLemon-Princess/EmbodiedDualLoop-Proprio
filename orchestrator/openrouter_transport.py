"""Small, action-free helpers for bounded OpenRouter HTTP recovery."""
import email.utils
import http.client
import math
import os
import ssl
import time
import urllib.error

DEADLINE_ENV = 'EMBODIEDEVO_SESSION_DEADLINE_MONOTONIC'


class TransportDeadlineExceeded(RuntimeError):
    """No further request or returned action may consume this expired budget."""


def session_deadline():
    value = os.environ.get(DEADLINE_ENV)
    if value is None:
        return None
    deadline = float(value)
    if not math.isfinite(deadline):
        raise ValueError('Session transport deadline must be finite')
    return deadline


def deadline_environment(deadline):
    environment = os.environ.copy()
    if deadline is not None:
        inherited = session_deadline()
        environment[DEADLINE_ENV] = str(min(deadline, inherited) if inherited is not None else deadline)
    return environment


def root_cause(error):
    while isinstance(error, urllib.error.URLError) and isinstance(error.reason, BaseException):
        error = error.reason
    return error


def retryable(error):
    status = error.code if isinstance(error, urllib.error.HTTPError) else getattr(error, 'status_code', None)
    if status is not None:
        return status in {408, 429} or 500 <= status <= 599
    cause = root_cause(error)
    if isinstance(cause, ssl.SSLCertVerificationError):
        return False
    if isinstance(cause, ssl.SSLError):
        return isinstance(cause, (ssl.SSLEOFError, ssl.SSLZeroReturnError)) or getattr(cause, 'reason', None) == 'UNEXPECTED_EOF_WHILE_READING'
    if isinstance(cause, (TimeoutError, ConnectionError, http.client.IncompleteRead)):
        return True
    # Some urllib/test transports retain only a textual EOF cause. Other URL
    # errors (bad proxy, DNS configuration, certificate validation) fail closed.
    return isinstance(cause, urllib.error.URLError) and 'eof' in str(cause.reason).lower() and 'certificate' not in str(cause.reason).lower()


def error_metadata(error):
    cause = root_cause(error)
    return {'error_type': type(error).__name__, 'root_cause_type': type(cause).__name__,
            'http_status': error.code if isinstance(error, urllib.error.HTTPError) else getattr(error, 'status_code', None),
            'errno': getattr(cause, 'errno', None), 'ssl_reason': getattr(cause, 'reason', None) if isinstance(cause, ssl.SSLError) else None}


def retry_after_seconds(headers, *, now=None):
    value = headers.get('Retry-After') if headers else None
    if value is None:
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            date = email.utils.parsedate_to_datetime(value)
            delay = date.timestamp() - (time.time() if now is None else now)
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0., delay) if math.isfinite(delay) else None


def remaining(deadline, *, now=None):
    value = deadline - (time.monotonic() if now is None else now)
    if value <= 0:
        raise TransportDeadlineExceeded('OpenRouter transport deadline exhausted; no returned action applied')
    return value
