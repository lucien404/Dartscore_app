import UIKit
let AKScreenWidth = UIScreen.main.bounds.size.width
let AKScreenHeight = UIScreen.main.bounds.size.height

class GameView: UIViewController, AKExcelViewDelegate {
    lazy var arrM = [Model]()
    lazy var arrScore = Int()
    lazy var excelView : AKExcelView = AKExcelView.init(frame: CGRect.init(x: 0, y: 180, width: AKScreenWidth, height: 140))
  @IBOutlet weak var segmented: UISegmentedControl!
  lazy var gameData = GameData.shared
    override func viewDidLoad() {
        super.viewDidLoad()
      view.backgroundColor = .white
      title = ""
      if #available(iOS 11.0, *) {
          
      }else{
          
          automaticallyAdjustsScrollViewInsets = false
      }

  
      // 自动滚到最近的一列
      excelView.autoScrollToNearItem = true
      // 设置表头背景色
      excelView.headerBackgroundColor = UIColor.cyan
      // 设置表头
      excelView.headerTitles = ["Player/Round","1th round","2th round","3th round","4th round","5th round","6th round","7th round","8th round","Total points"]
      // 设置间隙
      excelView.textMargin = 20
      // 设置左侧冻结栏数
      excelView.leftFreezeColumn = 1
      // 设置对应模型里面的属性  按顺序
      excelView.properties = ["playName","play1","play2","play3","play4","play5","play6","play7","play8","totle"]
      excelView.delegate = self
      // 指定列 设置 指定宽度  [column:width,...]
//      excelView.columnWidthSetting = [3:180]
      excelView.showNoDataView = true
      for i in 0 ..< 2 {
          let model = Model()
          model.playName = String.init("Player\(i+1):")
          model.play1 = String.init("")
          model.play2 = String.init("")
          model.play3 = String.init("")
          model.play4 = String.init("")
          model.play5 = String.init("")
          model.play6 = String.init("")
          model.play7 = String.init("")
          model.play8 = String.init("")
          model.totle = String.init("")
          arrM.append(model)
      }
      excelView.contentData = arrM
      view.addSubview(excelView)
      
//        excelView.reloadData()
//      setPlayer()
      initActorIndex()
      excelView.reloadDataCompleteHandler {
          print(" reload complete")

      }
    }
     
    override func viewDidAppear(_ animated: Bool){
        super.viewDidAppear(animated)
    }
  
  @IBAction func indexChanged(_ sender: Any) {
    gameData.setActorIndex(_index: segmented.selectedSegmentIndex)
  }
  func setPlayer() {
    let palyer1Data = gameData.getPlayer1Data()
    let palyer2Data = gameData.getPlayer2Data()
    
    let player1 = arrM[0]
    let player2 = arrM[1]
    for i in 0 ..< 8 {
      if i < palyer1Data.count  {
        let score = palyer1Data[i]
        player1.setPlayString(_index: i + 1, _str: String.init("\(score)"))
      }else
      {
        player1.setPlayString(_index: i + 1, _str: String.init(""))
      }
    }
    
    let totleScore1 = gameData.player1TotleScore()
    player1.setPlayerTotoleScore(_str: String.init("\(totleScore1)"))
    
    for i in 0 ..< 8 {
      if i < palyer2Data.count  {
        let score = palyer2Data[i]
        player2.setPlayString(_index: i + 1, _str: String.init("\(score)"))
        
//        if i == 7 {
//          let totleScore = gameData.player2TotleScore()
//          player2.setPlayerTotoleScore(_str: String.init("\(totleScore)"))
//        }
      }
      else
      {
        player2.setPlayString(_index: i + 1, _str: String.init(""))
      }
    }
    
    let totleScore2 = gameData.player2TotleScore()
    player2.setPlayerTotoleScore(_str: String.init("\(totleScore2)"))
    
    //如果都玩到第8次 则弹窗结算窗口
    if palyer2Data.count == 8 && palyer1Data.count == 8 {
      let totlePlayer2Score = gameData.player2TotleScore()
      let totlePlayer1Score = gameData.player1TotleScore()
      var str:String? = nil
      if totlePlayer1Score == totlePlayer2Score  {
        str = String.init("Player1 and Player2 draw")
      }else if totlePlayer1Score > totlePlayer2Score {
        str = String.init("Player1 is the Winner")
      }else
      {
        str = String.init("Player2 is the Winner")
      }
     
      DispatchQueue.main.async {
        DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 0.5) {
          let alertController = UIAlertController(title: str,
                                                  message: nil, preferredStyle: .alert)
          //显示提示框
          self.present(alertController, animated: true, completion: nil)
        }
        //两秒钟后自动消失
        DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 5) {
            self.presentedViewController?.dismiss(animated: false, completion: nil)
        }
      }
    
    }

    excelView.reloadDataCompleteExcelHandler {
        print(" reload complete")
//
    }
  }
  
  func TEST(_SCORE:Int) {
    let player1 = arrM[0]
    player1.setPlayString(_index: 1, _str: String.init("111111"))
//    excelView.contentData = arrM
    excelView.reloadDataCompleteExcelHandler {
        print(" reload complete")
//
    }
//
//          excelView.reloadData()
  }
  
  func initActorIndex() {
    let palyer1Data = gameData.getPlayer1Data()
//    let palyer2Data = gameData.getPlayer2Data()
    
    if palyer1Data.count == 8 {
      segmented.selectedSegmentIndex = 1
    }
    else
    {
      segmented.selectedSegmentIndex = 0
    }
    gameData.setActorIndex(_index: segmented.selectedSegmentIndex)
  }
  
  func showWinActor() {
    
  }
  
  @IBAction func restartGame(_ sender: Any) {
    self.dismiss(animated: true, completion: nil)
    gameData.reset()
    setPlayer()
    
    excelView.reloadDataCompleteExcelHandler {
        print(" reload complete")
//
    }
//    TEST(_SCORE:1000)
//    TEST(_SCORE: 1000)
  }
  
  @IBAction func settlement(_ sender: Any) {
  
      let totlePlayer2Score = gameData.player2TotleScore()
      let totlePlayer1Score = gameData.player1TotleScore()
      var str:String? = nil
      if totlePlayer1Score == totlePlayer2Score  {
        str = String.init("Draw")
      }else if totlePlayer1Score > totlePlayer2Score {
        str = String.init("Player1 Won")
      }else
      {
        str = String.init("Player2 Won")
      }
     
      DispatchQueue.main.async {
        DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 0.5) {
          let alertController = UIAlertController(title: str,
                                                  message: nil, preferredStyle: .alert)
          //显示提示框
          self.present(alertController, animated: true, completion: nil)
        }
        //两秒钟后自动消失
        DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 5) {
            self.presentedViewController?.dismiss(animated: false, completion: nil)
        }
      }
  }
  
  @IBAction func choicePic(_ sender: Any) {
    let controller: ViewController = ViewController()
    controller.setGameView(_gameView: self)
    present(controller, animated: true, completion: nil)
    controller.showImagePicker()
  }
  
  @IBAction func backToStartView(_ sender: Any) {
    self.dismiss(animated: true, completion: nil)
  }
  
  override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
}

