using UnityEngine;

public class ObjectController : MonoBehaviour {
    
    /// <summary>
    /// 初期化
    /// </summary>
    public void init () {
        gameObject.transform.rotation = Quaternion.identity;
        setActive(true);
    }
    
    /// <summary>
    /// 更新処理
    /// </summary>
    public void update () {
        gameObject.transform.Rotate(new Vector3(0,10,0)*Time.deltaTime);
    }

    /// <summary>
    /// オブジェクトのアクティブを変更
    /// </summary>
    /// <param name="enable">見えるか、見えないか</param>
    public void setActive(bool enable){
        gameObject.SetActive(enable);
    }
}