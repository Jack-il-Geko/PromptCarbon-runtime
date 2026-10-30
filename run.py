"""Start the local application after downloading missing pinned core assets."""
from runtime_assets import ensure_assets

if __name__=='__main__':
    ensure_assets()
    import server
    server.run()
