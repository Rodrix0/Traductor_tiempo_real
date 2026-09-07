import tempfile
import unittest
from pathlib import Path
from src.storage import HistoryStore
from src.subtitles import Caption


class StorageTests(unittest.TestCase):
    def test_persists_searches_and_exports_after_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'history.sqlite3'
            store = HistoryStore(path)
            key = store.create('película.mp4','archivo','en','pt','base')
            caption = Caption(1.25,4.5,'Hello friend','Olá amigo','en')
            store.add(key,caption)
            store.finish(key,'completado')
            other = store.create('Llamada','en vivo',None,'es','small')
            reopened = HistoryStore(path)
            self.assertEqual(reopened.captions(key), [caption])
            self.assertEqual(reopened.search('amigo')[0]['id'], key)
            self.assertEqual(reopened.search('película')[0]['count'], 1)
            self.assertEqual(reopened.search('amigo')[0]['state'], 'completado')
            self.assertEqual(reopened.search("'; DROP TABLE sessions; --"), [])
            self.assertEqual(len(reopened.search()), 2)
            self.assertEqual(reopened.captions(other), [])

    def test_partial_sessions_preserve_order(self):
        with tempfile.TemporaryDirectory() as folder:
            store = HistoryStore(Path(folder)/'history.sqlite3')
            key = store.create('cancelada','archivo','pt','es','base')
            for i in range(3):
                store.add(key, Caption(i,i+1,'Oi',f'Hola {i}','pt'))
            store.finish(key,'cancelado')
            self.assertEqual([c.start for c in store.captions(key)], [0,1,2])
            self.assertEqual(len(store.captions(key,limit=2)), 2)
            self.assertEqual(store.search()[0]['state'], 'cancelado')
