import os
import sys
import datetime
import shutil
from pathlib import Path
import yaml

# -------------------------- 可配置参数（请根据自身需求修改）--------------------------
CHECKPOINTS_DIR = Path("./checkpoints")  # 源根目录
A_ROOT_DIR = Path("./archive")                # 目标根目录
YAML_CONFIG_PATH = Path("./config.yaml") # YAML配置文件路径
TIME_FORMAT = "%y%m%d%H%M%S"              # 时间戳格式：YYMMDDHHmmss
# -----------------------------------------------------------------------------------


def get_single_key():
    """跨平台监听单次按键，无需回车，返回小写字符"""
    if sys.platform == "win32":
        import msvcrt
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getch().decode('utf-8').lower()
                return char
    else:
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty = termios.tcgetattr(fd)
            tty[3] &= ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, tty)
            char = sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return char


def read_yaml_title(yaml_path: Path) -> str:
    """读取YAML的title字段，过滤文件名非法字符，兼容跨平台创建目录"""
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML配置文件不存在：{yaml_path}")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        title = config.get("model").get("type")
        if not title:
            raise ValueError("YAML文件中未找到有效的 title 配置项")
        
        # 过滤Windows/Linux非法文件名字符
        invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
        title = str(title).strip()
        for char in invalid_chars:
            title = title.replace(char, '_')
        return title
    except yaml.YAMLError as e:
        raise RuntimeError(f"YAML文件解析失败：{e}") from e


def move_checkpoints_files():
    """递归迁移：移动checkpoints下所有文件、子目录至新文件夹，复制YAML配置"""
    # 校验源目录
    if not CHECKPOINTS_DIR.is_dir():
        print(f"错误：源目录 {CHECKPOINTS_DIR} 不存在，操作终止")
        return

    # 读取配置标题
    try:
        title = read_yaml_title(YAML_CONFIG_PATH)
        print(f"成功读取配置标题：{title}")
    except Exception as e:
        print(f"读取配置失败：{e}")
        return

    # 生成目标目录路径
    timestamp = datetime.datetime.now().strftime(TIME_FORMAT)
    target_folder_name = f"{title}_{timestamp}"
    target_dir = A_ROOT_DIR / target_folder_name

    # 创建目标目录（支持递归创建）
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"成功创建目标目录：{target_dir.resolve()}")
    except Exception as e:
        print(f"创建目录失败：{e}")
        return

    # ========== 递归移动：迁移所有文件/子目录（核心修改部分）==========
    success_count = 0
    fail_count = 0
    # 遍历checkpoints根目录下所有顶层条目（文件/文件夹）
    for item in CHECKPOINTS_DIR.iterdir():
        target_path = target_dir / item.name
        try:
            shutil.move(str(item), str(target_path))
            print(f"已迁移：{item}")
            success_count += 1
        except Exception as e:
            print(f"迁移失败 {item}，原因：{str(e)}")
            fail_count += 1

    # 复制YAML配置文件至新目录
    try:
        yaml_dest = target_dir / YAML_CONFIG_PATH.name
        shutil.copy2(YAML_CONFIG_PATH, yaml_dest)
        print(f"\n配置文件已复制至：{yaml_dest}")
    except Exception as e:
        print(f"\n复制配置文件失败：{str(e)}")

    # 输出执行结果
    print("\n===== 文件迁移操作完成 =====")
    print(f"成功迁移条目数：{success_count}")
    print(f"迁移失败条目数：{fail_count}")
    print(f"所有文件存储路径：{target_dir.resolve()}")


def delete_checkpoints_files():
    """递归删除：清空checkpoints下所有文件、子目录，保留根目录本身"""
    # 单次安全确认
    confirm = input("警告：将递归删除所有文件/子目录！输入 [Y/y] 确认删除：")
    if confirm.strip().lower() != "y":
        print("删除操作已取消，程序退出")
        return

    # 校验目录存在
    if not CHECKPOINTS_DIR.is_dir():
        print(f"错误：目录 {CHECKPOINTS_DIR} 不存在，无需删除")
        return

    # ========== 递归删除：清空所有内容（核心修改部分）==========
    success_count = 0
    fail_count = 0
    for item in CHECKPOINTS_DIR.iterdir():
        try:
            if item.is_file():
                # 删除普通文件
                item.unlink()
            elif item.is_dir():
                # 递归删除整个子目录
                shutil.rmtree(item)
            print(f"已删除：{item}")
            success_count += 1
        except Exception as e:
            print(f"删除失败 {item}，原因：{str(e)}")
            fail_count += 1

    # 输出执行结果
    print("\n===== 文件删除操作完成 =====")
    print(f"成功删除条目数：{success_count}")
    print(f"删除失败条目数：{fail_count}")
    print(f"备注：{CHECKPOINTS_DIR} 根目录已保留")


def main():
    """主程序：监听按键并分发操作逻辑"""
    print("=" * 60)
    print("程序已启动，支持递归操作所有文件/子目录")
    print("T/t - 递归迁移 checkpoints 所有内容至新目录")
    print("D/d - 递归清空 checkpoints 所有内容（保留目录）")
    print("=" * 60)

    # 监听有效按键
    while True:
        key = get_single_key()
        if key == "t":
            print("\n>>> 触发【递归迁移】操作 <<<\n")
            move_checkpoints_files()
            break
        elif key == "d":
            print("\n>>> 触发【递归删除】操作 <<<\n")
            delete_checkpoints_files()
            break
        else:
            print(f"\n无效按键 [{key}]，请按 T/t 或 D/d")


if __name__ == "__main__":
    main()
    print("\n程序已正常退出")