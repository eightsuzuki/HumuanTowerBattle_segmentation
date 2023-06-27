using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Linq;
using System.Threading;
using UnityEngine.SceneManagement;


public class CreateManager : MonoBehaviour
{
    private GameObject obj;
    public List<GameObject> people;//どうぶつ取得配列
    public bool isFall;
    int file_length;
    public float pivotHeight = 3;//生成位置の基準
    public Camera mainCamera;//カメラ取得用変数
    public GameObject cameracontroller;
    // Start is called before the first frame update
    void Init()
    {
        Animal.isMoves.Clear();//移動してる動物のリストを初期化
        string[] files = Directory.GetFiles(
            @"Assets/Resources", "*.png", SearchOption.AllDirectories
            ).ToArray();
        file_length = files.Length;
 //       obj = null;
    }

    void Start()
    {
        
        string[] files = Directory.GetFiles(
              @"Assets/Resources", "*.png", SearchOption.AllDirectories
              );
        foreach (string file in files)
        {
            File.SetAttributes(file, FileAttributes.Normal);
            File.Delete(file);
        }
        Init();
    }

    // Update is called once per frame
    void Update()
    {
//        if (obj && obj.transform.position.y < -5)
//        {
//        }
        if (CheckGameOver(people)){
          SceneManager.LoadScene ("GameOver");
        }
        if (CheckMove(Animal.isMoves))
        {
            return;//移動中なら処理はここまで
        }
        string[] files = Directory.GetFiles(
            @"Assets/Resources", "*.png", SearchOption.AllDirectories
            ).OrderBy(f => File.GetLastWriteTime(f).Date
            ).ToArray();
        if (files.Length == 0){
            return;
        }
        if (files.Length > file_length)
        {
            string tar = files[files.Length - 1].Remove(0, 17);
            tar = tar.Replace(".png", "");
            Sprite img = Resources.Load(tar, typeof(Sprite)) as Sprite;
//            Debug.Log(img);
            if (img == null){
                return;
            }
            Create(img);
            file_length += 1;
        }
        /*Vector2 v = new Vector2(mainCamera.ScreenToWorldPoint(Input.mousePosition).x, pivotHeight);

        if (Input.GetMouseButtonUp(0))//もし（マウス左クリックが離されたら）
        {
            if (!RotateButton.onButtonDown)//ボタンをクリックしていたら反応させない
            {
                obj.transform.position = v;
                obj.GetComponent<Rigidbody2D>().isKinematic = false;//――――物理挙動・オン
                isFall = true;//落ちて、どうぞ
            }
            RotateButton.onButtonDown = false;//マウスが上がったらボタンも離れたと思う
        }
        else if (Input.GetMouseButton(0))//ボタンが押されている間
        {
            obj.transform.position = v;
        }*/
    }

    void Create(Sprite img)
    {
        /*while (CameraController.isCollision)
        {
            Debug.Log("collision_start");
            cameracontroller.transform.Translate(0, 2.0f, 0);
            mainCamera.transform.Translate(0, 2.0f, 0);//カメラを少し上に移動
            pivotHeight += 2.0f;//生成位置も少し上に移動
        Debug.Log("collision_fin");
        }*/
        if (CameraController.isCollision)
        {
            Debug.Log("collision_start");
            cameracontroller.transform.Translate(0, 3.0f, 0);
            mainCamera.transform.Translate(0, 3.0f, 0);
            pivotHeight += 3.0f;
            Debug.Log("collision_fin");
        }
        isFall = false;
        obj = new GameObject();
        obj.AddComponent<SpriteRenderer>();
        obj.GetComponent<SpriteRenderer>().sprite = img;
        obj.AddComponent<PolygonCollider2D>();
        obj.AddComponent<Rigidbody2D>();
        //obj.GetComponent<Rigidbody2D>().isKinematic = true;
        obj.AddComponent<Animal>();
        obj.transform.position = new Vector3(0.0f, pivotHeight, 0.0f);
        people.Add(obj);
    }

    /// <summary>
    /// 移動中かチェック
    /// </summary>
    /// <param name="isMoves"></param>
    /// <returns></returns>
    bool CheckMove(List<Moving> isMoves)
    {
        if (isMoves == null)
        {
            return false;
        }
        foreach (Moving b in isMoves)
        {
            if (b.isMove)
            {
                //Debug.Log("移動中(*'ω'*)");
                return true;
            }
        }
        return false;
    }

    bool CheckGameOver(List<GameObject> people)
    {
        foreach (GameObject b in people)
        {
            if (b.transform.position.y < -5)
            {
                return true;
            }
        }
        return false;
    }
    
    /// <summary>
    /// どうぶつの回転
    /// ボタンにつけて使います
    /// </summary>
    public void RotateAnimal()
    {
        if (!isFall)
        {
            obj.transform.Rotate(0, 0, -30);//30度ずつ回転
        }
    }
}
