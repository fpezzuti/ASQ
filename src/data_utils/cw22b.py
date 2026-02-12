import os
import gzip
import json
from io import BytesIO
from typing import Tuple, Dict, Iterator, Optional
from tqdm import tqdm


def get_corpus_iter(start_txt_dir: str, verbose: bool = True, cat_docids: bool = True, ret_url=False) -> Iterator[Dict]:
    count = 0
    with tqdm(total=0, unit=' docs', unit_scale=True, desc='Processing documents', disable=not verbose) as pbar:
        for dirname in sorted(os.listdir(start_txt_dir)): # e.g. 'en00', 'en01', ...
            subdir_path = os.path.join(start_txt_dir, dirname)
            if not os.path.isdir(subdir_path):
                continue
            if verbose:
                print(f"Processing subdir {subdir_path}.")
            for subdirname in sorted(os.listdir(subdir_path)):  # e.g. 'en0000-00099', ...
                subsubdir_path = os.path.join(subdir_path, subdirname)
                if not os.path.isdir(subsubdir_path):
                    continue
                if verbose:
                    print(f"Processing subsubdir {subsubdir_path}.")
                for filename in sorted(os.listdir(subsubdir_path)):
                    if filename.endswith('.json.gz'):
                        file_path = os.path.join(subsubdir_path, filename)
                    
                        for doc_data in read_json_gz(file_path):
                            docid = get_docid_from_doc_data(doc_data)
                            if cat_docids:
                                docid = cat_docid(docid)
                            text = get_text_from_doc_data(doc_data)
                            title = get_title_from_doc_data(doc_data)
                            if docid is None or text is None or title is None:
                                print("Skipping document due to missing fields:", doc_data)
                                continue
                            if ret_url:
                                url = get_url_from_doc_data(doc_data)
                                yield {'docno': docid,
                                       'text': text,
                                       'title':  title,
                                       'url': url}
                            else:
                                yield {'docno': docid,
                                    'text': text,
                                    'title':  title}
                                
                            count += 1
                            pbar.update(1)

                            if verbose and count % 10_000_000 == 0:
                                print(f"Processed {count} documents so far.")

        if verbose:
            print(f"Completed {subdir_path}, current total: {count}")

def cat_docid(docid: str, to_str=True) -> str | int:
    prefix = 'clueweb22-en' 
    cat_docid = docid[len(prefix):]
    cat_docid = cat_docid.replace('-', '')  # remove spaces
    if to_str:
        return cat_docid
    else:
        return int(cat_docid)

def decat_docid(docid_cat: str | int) -> str:
    """
        Reconstructs the document id by adding prefix and hyphens.
    """
    if isinstance(docid_cat, int):
        docid_cat = str(docid_cat).zfill(11)  # pad with leading zeros to ensure length 11
    docid = "clueweb22-en" + docid_cat[:4] + '-' + docid_cat[4:6] + '-' + docid_cat[6:11]
    return docid

def get_docid_from_doc_data(doc_data: Dict) -> str:
    """
        Extracts the document id from the document data dictionary.
    """
    DOCID_KEY = 'ClueWeb22-ID'
    docid = doc_data[DOCID_KEY]
    return docid

def get_text_from_doc_data(doc_data: Dict) -> str:
    """
        Extracts the text from the document data dictionary.
    """
    TEXT_KEY = "Clean-Text"
    text = doc_data.get(TEXT_KEY, None)
    return text.strip() if text else None

def get_title_from_doc_data(doc_data: Dict) -> Optional[str]:
    """
        Extracts the title from the document data dictionary.
    """
    TITLE_KEY = "Title"
    title = doc_data.get(TITLE_KEY, "")
    return title

def get_url_from_doc_data(doc_data: Dict) -> Optional[str]:
    """
        Extracts the URL from the document data dictionary.
    """
    URL_KEY = "URL"
    url = doc_data.get(URL_KEY, None)
    return url.strip() if url else None


def read_json_gz(file_path: str) -> Iterator[Dict]:
    """
        Yields line by line a JSON object from a gzipped file (avoids loading the entire file in memory).

        Args: 
            file_path: path to the gzipped file
    """
    json_loads: callable = json.loads  # local alias for performance
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue  # skip empty lines
            try:
                yield json_loads(stripped_line)  # parse JSON obj.
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {stripped_line}\n{e}")
                continue  # skip lines that cannot be decoded

def decompress_gzip_data(compressed_data: bytes) -> bytes:
    """
        Decompresses a gzip compressed data bytes.
    """
    with gzip.GzipFile(fileobj=BytesIO(compressed_data), mode='rb') as f:
        return f.read()


def random_read_json_gz(file_path: str, start_offset: int, end_offset: int | None = None) -> Dict:
    """
        Read a JSON object from a gzipped file at a random offset.
        
        Args:
            file_path: path to the gzipped file
            start_offset: start offset to read from
            end_offset: end offset to read until
    """
    json_loads = json.loads  # local alias for performance
    with open(file_path, 'rb') as gz_file:
        gz_file.seek(start_offset)
        read_length = None if end_offset is None else (end_offset - start_offset)
        compressed_document = gz_file.read(read_length)
        try:
            decompressed_line = decompress_gzip_data(compressed_document).decode('utf-8') 
            return json_loads(decompressed_line)
        except json.JSONDecodeError as e:
            return None   

def read_offsets(file_path: str, i: int) -> Tuple[int, int]:
    """
        Read offsets from a file and returns a tuple storing the start and end offsets.

        Args:
            file_path: path to the file containing offsets
            i: index of the offset to read

    """
    OFFSET_DIM = 10
    offset_size = OFFSET_DIM + 1  # 10 digits + newline
    with open(file_path, 'r') as f:
        f.seek(i * offset_size)
        start_offset = f.read(OFFSET_DIM)
        end_offset = f.read(OFFSET_DIM) if f.read(1) else None

    if start_offset == '' or end_offset == '':
        print(f"Error reading offsets for index {i} in file {file_path}.")
        return None, None
    return int(start_offset), int(end_offset)


def get_parts(docid: str) -> Tuple[str, str, int]:
    """
        Get file path's parts given a clueweb id.
    """
    _, subsubdir, file_seq, doc_seq = docid.split('-')
    subdir = subsubdir[0:4]  # e.g., "en00", "en01", etc.
    return subdir, subsubdir, file_seq, int(doc_seq)

def reconstruct_docid(fpath: str, doc_seq: int) -> str:
    """
        Reconstruct the clueweb document id from a file path and document sequence number.
        Args:
            fpath: path to the file containing the document
            doc_seq: document sequence number
    """
    filename = os.path.basename(fpath)
    prefix = filename.split('.', 2)[0]
    return f"Clueweb22-{prefix}-{doc_seq:05d}"


def navigate_to_id(dir_path: str, docid: str, ret='data', cat_docid: bool = False) -> Dict:
    """
        Navigate to the directory where the data for a document given clueweb id is stored.

        Args:
            dir_path: path to the directory (inlinks, outlinks, txt, etc.)
            docid: clueweb document id
            ret: what to return ('data' for document data, 'text' for offsets text, 'all' for text, title, url)
    """
    if cat_docid:
        cw22_docid = decat_docid(docid)
    else:
        cw22_docid = docid
    subdir, subsubdir, file_seq, doc_seq = get_parts(cw22_docid)

    file_prefix = os.path.join(f"{dir_path}{subdir}/", subsubdir, f"{subsubdir}-{file_seq}")
    file_fpath = f"{file_prefix}.json.gz"
    offsets_fpath = f"{file_prefix}.offset"
    
    try:
        start_offset, end_offset = read_offsets(offsets_fpath, doc_seq)
        if start_offset is None and end_offset is None:
            return None # no data for this docid
    except Exception as e:
        raise RuntimeError(f"Error reading 'start' and 'end' offsets from file={offsets_fpath}: {e}")

    try:
        doc_data = random_read_json_gz(file_fpath, start_offset, end_offset)
        #print("Doc data keys:", list(doc_data.keys()))
        if ret == 'text':
            doc_text = get_text_from_doc_data(doc_data)
            return doc_text
        elif ret == 'all':
            doc_text = get_text_from_doc_data(doc_data)
            doc_title = get_title_from_doc_data(doc_data)
            doc_url = get_url_from_doc_data(doc_data)
            return {
                'text': doc_text,
                'title': doc_title,
                'url': doc_url
            }
        elif ret == 'data':
            return doc_data
        else:
            raise ValueError(f"Invalid return type specified: {ret}")
    except Exception as e:
        raise RuntimeError(f"Error reading from gzipped JSON file={file_fpath}: {e}")
    


def main(): # tests
    dir_path = "./cw22b/txt/en/"
    docid = "clueweb22-en0011-49-01114"
    print("Os.path exists:", os.path.exists("./cw22b/txt/en/en00/en0011/en0011-49.offset"))
    doc_data = navigate_to_id(dir_path, docid)
    print("Keys in doc_data:", list(doc_data.keys()))
    print("Text:", get_text_from_doc_data(doc_data)[:500])
    print("Title:", get_title_from_doc_data(doc_data))
    print("URL:", get_url_from_doc_data(doc_data))
    print("Docid:", get_docid_from_doc_data(doc_data))
    print("Original docid:", decat_docid(cat_docid(get_docid_from_doc_data(doc_data))))
    assert get_docid_from_doc_data(doc_data) == docid, "Docid mismatch!"
    assert decat_docid(cat_docid(get_docid_from_doc_data(doc_data))) == docid, "Docid reconstruction mismatch!"



if __name__ == "__main__":
    main()