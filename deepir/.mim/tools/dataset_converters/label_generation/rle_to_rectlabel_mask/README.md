Label Studio 会将我们的语义分割标注以 raw JSON 格式储存在 SQLite（默认）、PostgreSQL 等数据库后端。
如果直接导出 PNG 图像，这些图像的文件名将会类似于 `task-1-annotation-1-by-1-tag-sky-0.png`，丧失原图像名，无法与标注数据对应。
因此，我们需要从 raw JSON 中读取储存标记结果的 RLE 格式的数据，将其转换为 RectLabel 能够读取的格式，即 Mask Image 与 XML。

流程如下：

1. 将 RLE 转化为 Mask Image：运行 `rle_to_binary_mask.py`，将会在输出文件夹下生成 Mask Image。
2. 为每幅图像生成对应的 XML：运行 `gen_mask_xml.py`，将会在输出文件夹下生成 XML。
3. 将 Mask Image 与 XML 放一起，即可在 RectLabel 中查看并修正标注结果。