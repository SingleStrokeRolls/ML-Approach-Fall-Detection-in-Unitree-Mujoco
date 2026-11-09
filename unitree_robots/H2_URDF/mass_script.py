import xml.etree.ElementTree as ET
import numpy as np

def get_robot_mass(xml_file):
    """
    从MuJoCo XML文件中提取机器人总质量
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    total_mass = 0.0
    body_masses = []
    
    # 遍历所有body元素
    for body in root.iter('body'):
        # 查找inertial子元素
        inertial = body.find('inertial')
        if inertial is not None:
            mass = inertial.get('mass')
            if mass is not None:
                mass_val = float(mass)
                total_mass += mass_val
                body_masses.append({
                    'name': body.get('name', 'unnamed'),
                    'mass': mass_val
                })
    
    return total_mass, body_masses

def print_mass_breakdown(xml_file):
    """
    打印详细的重量分布
    """
    total_mass, body_masses = get_robot_mass(xml_file)
    
    print("=" * 50)
    print("H2机器人质量分析报告")
    print("=" * 50)
    print(f"\n总质量: {total_mass:.4f} kg ({total_mass:.2f} kg)")
    print(f"\n各连杆质量分布:")
    print("-" * 50)
    
    # 按质量排序
    body_masses.sort(key=lambda x: x['mass'], reverse=True)
    
    for item in body_masses:
        percentage = (item['mass'] / total_mass) * 100
        print(f"{item['name']:35s} {item['mass']:8.4f} kg ({percentage:5.2f}%)")
    
    print("-" * 50)
    print(f"{'总计':35s} {total_mass:8.4f} kg (100.00%)")
    print("=" * 50)
    
    return total_mass

# 主程序
if __name__ == "__main__":
    xml_file = "H2_modified.xml"  # 修改为你的XML文件名
    
    try:
        total = print_mass_breakdown(xml_file)
        
        # 额外统计
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 按部位分组统计
        groups = {
            '躯干/腰部': ['pelvis', 'waist', 'torso'],
            '头部': ['head'],
            '左臂': ['left_shoulder', 'left_elbow', 'left_wrist'],
            '右臂': ['right_shoulder', 'right_elbow', 'right_wrist'],
            '左腿': ['left_hip', 'left_knee', 'left_ankle'],
            '右腿': ['right_hip', 'right_knee', 'right_ankle']
        }
        
        print("\n按部位分组统计:")
        print("-" * 50)
        
        for group_name, keywords in groups.items():
            group_mass = 0.0
            for body in root.iter('body'):
                body_name = body.get('name', '')
                inertial = body.find('inertial')
                if inertial is not None:
                    if any(kw in body_name for kw in keywords):
                        mass = float(inertial.get('mass', 0))
                        group_mass += mass
            
            percentage = (group_mass / total) * 100
            print(f"{group_name:15s}: {group_mass:7.2f} kg ({percentage:5.2f}%)")
        
        print("=" * 50)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{xml_file}'")
    except Exception as e:
        print(f"错误: {e}")