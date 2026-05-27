# Bundled Image Fonts

The image renderer only uses bundled fonts in this directory. Users can choose
one of these built-in families from plugin configuration, but arbitrary system
fonts and custom font paths are intentionally not exposed.

| Config option | Internal key | Regular file | Bold/title file |
| --- | --- | --- | --- |
| 默认黑体 | `noto_sans` | `NotoSansCJKsc-Regular.otf` | `NotoSansCJKsc-Bold.otf` |
| 专业宋体 | `noto_serif` | `NotoSerifCJKsc-Regular.otf` | `NotoSerifCJKsc-Bold.otf` |
| 清新文楷 | `lxgw_wenkai` | `LXGWWenKai-Regular.ttf` | `LXGWWenKai-Medium.ttf` |
| 标题手札 | `zcool_xiaowei` | `ZCOOLXiaoWei-Regular.ttf` | `ZCOOLXiaoWei-Regular.ttf` |
| 科技窄体 | `zcool_qingke` | `ZCOOLQingKeHuangYou-Regular.ttf` | `ZCOOLQingKeHuangYou-Regular.ttf` |

License files for the bundled open fonts are kept next to the font files.
