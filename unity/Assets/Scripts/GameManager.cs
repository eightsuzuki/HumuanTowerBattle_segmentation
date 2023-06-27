using UnityEngine;

public class GameManager : MonoBehaviour {

    public ObjectController red, green, blue;

    //シーン番号
    int step;
    int next_step = -1;

    /* 書ける人はこっちのほうがいい
    enum STATE
    {
        NONE = -1,
        RED,
        GREEN,
        BLUE,
    }
    STATE step;
    STATE next_step;
    */

    void Start () {
        next_step = 0;
    }
    
    void Update () {
        //次のステートチェック
        if(next_step != -1)
        {
            switch (next_step)
            {
                case 0://赤初期化
                    red.init();
                    green.setActive(false);
                    blue.setActive(false);
                    break;
                case 1://緑初期化
                    red.setActive(false);
                    green.init();
                    blue.setActive(false);
                    break;
                case 2://青初期化
                    red.setActive(false);
                    green.setActive(false);
                    blue.init();
                    break;
            }
            //ここはとても大事！
            step = next_step;
            next_step = -1;
        }
        //現在ステートの更新
        switch (step)
        {
            case 0://赤更新
                if (Input.GetMouseButtonUp(0))
                {
                    next_step = 1;
                }
                red.update();
                break;
            case 1://緑更新
                if (Input.GetMouseButtonUp(0))
                {
                    next_step = 2;
                }
                green.update();
                break;
            case 2://青更新
                if (Input.GetMouseButtonUp(0))
                {
                    next_step = 0;
                }
                blue.update();
                break;
        }
    }
}