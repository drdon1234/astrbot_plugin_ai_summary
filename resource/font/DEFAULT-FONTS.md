# Runtime Image Fonts

Font binaries are not included in the plugin package. During plugin
initialization and before image rendering, the plugin downloads missing or
invalid files from the `drdon1234/fonts` repository's immutable `v1.0.0`
release. Each file is checked against its expected byte size and SHA-256 before
it atomically replaces the local copy in this directory.

Users can choose one of these managed families from plugin configuration, but
arbitrary system fonts and custom font paths are intentionally not exposed.

| Config option | Internal key | Regular file | Bold/title file |
| --- | --- | --- | --- |
| 默认黑体 | `noto_sans` | `NotoSansCJKsc-Regular.otf` | `NotoSansCJKsc-Bold.otf` |
| 专业宋体 | `noto_serif` | `NotoSerifCJKsc-Regular.otf` | `NotoSerifCJKsc-Bold.otf` |
| 清新文楷 | `lxgw_wenkai` | `LXGWWenKai-Regular.ttf` | `LXGWWenKai-Medium.ttf` |
| 标题手札 | `zcool_xiaowei` | `ZCOOLXiaoWei-Regular.ttf` | `ZCOOLXiaoWei-Regular.ttf` |
| 科技窄体 | `zcool_qingke` | `ZCOOLQingKeHuangYou-Regular.ttf` | `ZCOOLQingKeHuangYou-Regular.ttf` |

License files for these open fonts remain next to this notice. The same license
files accompany the downloadable font assets in the font repository.
