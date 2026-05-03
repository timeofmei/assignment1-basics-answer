# WriteUp

## 2.1

### (a)

`chr(0)` returns `\x00`.

### (b)

```python
>>> print(chr(0))

>>> chr(0).__repr__()
"'\\x00'"
```
### (c)

It will be ignored in the `print` output.

## 2.2

### (a)

UTF-8 encoded byte sequence is the shortest. 

```python
>>> utf16_encoded = test_string.encode("utf-16")
>>> utf32_encoded = test_string.encode("utf-32")
>>> len(utf_encoded)
23
>>> len(utf16_encoded)
28
>>> len(utf32_encoded)
56
```

### (b)

Some characters are represented in multiple bytes, so any byte alone in its representation do not mean anything.

```python
>>> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
...     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
... 
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
>>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8")) 
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```

### (c)

```python
>>> b'\xff\xfe'.decode("utf-16")
''
>>> b'\xff\xfe'.decode("utf-8") 
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```
`\xff\xfe` is an example.

